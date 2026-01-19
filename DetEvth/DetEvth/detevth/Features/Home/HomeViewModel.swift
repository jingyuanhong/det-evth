import Foundation
import Combine

// Model for displaying screening results
struct ScreeningResultModel: Identifiable {
    let id: UUID
    let date: Date
    let primaryCondition: String
    let confidence: Float
    let source: String
    var signal: [Float]?  // ECG signal data for waveform display
    var probabilities: [Float]?  // All disease probabilities
    var heartRate: Int?  // Extracted heart rate from ECG
}

@MainActor
class HomeViewModel: ObservableObject {
    @Published var averageHeartRate: Int?
    @Published var minHeartRate: Int?
    @Published var maxHeartRate: Int?
    @Published var recentScreenings: [ScreeningResultModel] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private var cancellables = Set<AnyCancellable>()
    private let repository = ScreeningRepository.shared
    private let healthKitService = HealthKitService.shared

    func loadData() {
        loadHeartRateData()
        loadRecentScreenings()
    }

    private func loadHeartRateData() {
        // Try to fetch from HealthKit
        Task {
            var healthKitHasData = false

            do {
                // Request authorization if needed
                if !healthKitService.isAuthorized {
                    try await healthKitService.requestAuthorization()
                }

                // Get today's heart rate stats
                let stats = try await healthKitService.calculateHeartRateStats(days: 1)

                await MainActor.run {
                    if stats.average > 0 {
                        self.averageHeartRate = Int(stats.average)
                        self.minHeartRate = Int(stats.min)
                        self.maxHeartRate = Int(stats.max)
                        healthKitHasData = true
                    }
                }
            } catch {
                print("[HomeViewModel] HealthKit error: \(error.localizedDescription)")
            }

            // Fallback: If no HealthKit data, use heart rate from most recent ECG
            if !healthKitHasData {
                await MainActor.run {
                    self.loadHeartRateFromRecentECG()
                }
            }
        }
    }

    /// Fallback: Extract heart rate from the most recent ECG screening
    private func loadHeartRateFromRecentECG() {
        let entities = repository.fetchRecentScreeningResults(limit: 1)

        if let latestEntity = entities.first,
           let signal = latestEntity.ecgRecord?.signal,
           !signal.isEmpty {
            let sampleRate = latestEntity.ecgRecord?.sampleRate ?? 500
            if let heartRate = ECGHeartRateExtractor.extractHeartRate(from: signal, sampleRate: sampleRate) {
                self.averageHeartRate = heartRate
                self.minHeartRate = nil  // Not available from single ECG
                self.maxHeartRate = nil
                print("[HomeViewModel] Using heart rate from recent ECG: \(heartRate) bpm")
                return
            }
        }

        // No data available at all
        self.averageHeartRate = nil
        self.minHeartRate = nil
        self.maxHeartRate = nil
    }

    private func loadRecentScreenings() {
        // Fetch recent screenings from Core Data
        let entities = repository.fetchRecentScreeningResults(limit: 5)
        recentScreenings = entities.map { entity in
            // Try to extract heart rate from signal if available
            var heartRate: Int? = nil
            if let signal = entity.ecgRecord?.signal, !signal.isEmpty {
                let sampleRate = entity.ecgRecord?.sampleRate ?? 500
                heartRate = ECGHeartRateExtractor.extractHeartRate(from: signal, sampleRate: sampleRate)
            }

            return ScreeningResultModel(
                id: entity.id ?? UUID(),
                date: entity.createdAt ?? Date(),
                primaryCondition: entity.primaryCondition?.nameEN ?? "Unknown",
                confidence: entity.primaryConfidence,
                source: entity.ecgRecord?.source ?? "import",
                signal: entity.ecgRecord?.signal,
                probabilities: entity.probabilities.isEmpty ? nil : entity.probabilities,
                heartRate: heartRate
            )
        }
    }
}

// MARK: - ECG Heart Rate Extractor

/// Extracts heart rate from ECG signal by detecting R-peaks
struct ECGHeartRateExtractor {

    /// Extract heart rate from ECG signal
    /// - Parameters:
    ///   - signal: ECG signal values
    ///   - sampleRate: Sample rate in Hz
    /// - Returns: Heart rate in bpm, or nil if cannot be determined
    static func extractHeartRate(from signal: [Float], sampleRate: Float) -> Int? {
        guard signal.count > Int(sampleRate * 2) else {
            return nil  // Need at least 2 seconds of data
        }

        // Find R-peaks using simple threshold-based detection
        let rPeaks = detectRPeaks(signal: signal, sampleRate: sampleRate)

        guard rPeaks.count >= 2 else {
            return nil  // Need at least 2 R-peaks
        }

        // Calculate RR intervals
        var rrIntervals: [Float] = []
        for i in 1..<rPeaks.count {
            let interval = Float(rPeaks[i] - rPeaks[i-1]) / sampleRate
            // Filter out unrealistic intervals (HR between 30-200 bpm)
            if interval > 0.3 && interval < 2.0 {
                rrIntervals.append(interval)
            }
        }

        guard !rrIntervals.isEmpty else {
            return nil
        }

        // Calculate average RR interval and convert to BPM
        let avgRR = rrIntervals.reduce(0, +) / Float(rrIntervals.count)
        let heartRate = 60.0 / avgRR

        // Validate reasonable heart rate range
        if heartRate >= 30 && heartRate <= 200 {
            return Int(heartRate.rounded())
        }

        return nil
    }

    /// Detect R-peaks in ECG signal
    private static func detectRPeaks(signal: [Float], sampleRate: Float) -> [Int] {
        var peaks: [Int] = []

        // Calculate signal statistics
        let mean = signal.reduce(0, +) / Float(signal.count)
        let variance = signal.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(signal.count)
        let stdDev = sqrt(variance)

        // Threshold for R-peak detection (typically R-peaks are > 1.5 std above mean)
        let threshold = mean + 1.5 * stdDev

        // Minimum distance between peaks (refractory period ~200ms)
        let minDistance = Int(sampleRate * 0.2)

        var lastPeakIndex = -minDistance

        // Simple peak detection
        for i in 2..<(signal.count - 2) {
            // Check if this is a local maximum above threshold
            if signal[i] > threshold &&
               signal[i] > signal[i-1] &&
               signal[i] > signal[i-2] &&
               signal[i] > signal[i+1] &&
               signal[i] > signal[i+2] &&
               (i - lastPeakIndex) >= minDistance {

                peaks.append(i)
                lastPeakIndex = i
            }
        }

        return peaks
    }
}
