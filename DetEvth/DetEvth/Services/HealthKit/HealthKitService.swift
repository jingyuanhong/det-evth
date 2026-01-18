// HealthKitService.swift
// HealthKit Integration for ECG and Heart Rate Data
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import HealthKit
import Combine

/// Service for reading ECG and heart rate data from HealthKit
final class HealthKitService: ObservableObject {

    // MARK: - Singleton

    static let shared = HealthKitService()

    // MARK: - Properties

    private let healthStore = HKHealthStore()
    private var cancellables = Set<AnyCancellable>()

    @Published var isAuthorized = false
    @Published var authorizationError: Error?
    @Published var latestHeartRate: Double?
    @Published var ecgRecordings: [HKElectrocardiogram] = []

    // Types we need to read
    private let ecgType = HKObjectType.electrocardiogramType()
    private let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!

    // MARK: - Initialization

    private init() {
        checkAuthorizationStatus()
    }

    // MARK: - Authorization

    /// Check if HealthKit is available on this device
    var isHealthKitAvailable: Bool {
        HKHealthStore.isHealthDataAvailable()
    }

    /// Check current authorization status
    func checkAuthorizationStatus() {
        guard isHealthKitAvailable else {
            isAuthorized = false
            return
        }

        let ecgStatus = healthStore.authorizationStatus(for: ecgType)
        let heartRateStatus = healthStore.authorizationStatus(for: heartRateType)

        isAuthorized = ecgStatus == .sharingAuthorized || heartRateStatus == .sharingAuthorized
    }

    /// Request authorization to read ECG and heart rate data
    func requestAuthorization() async throws {
        guard isHealthKitAvailable else {
            throw HealthKitError.notAvailable
        }

        let typesToRead: Set<HKObjectType> = [
            ecgType,
            heartRateType
        ]

        try await healthStore.requestAuthorization(toShare: [], read: typesToRead)

        await MainActor.run {
            checkAuthorizationStatus()
        }
    }

    // MARK: - ECG Data

    /// Fetch ECG recordings from HealthKit
    /// - Parameters:
    ///   - limit: Maximum number of recordings to fetch
    ///   - startDate: Optional start date filter
    /// - Returns: Array of ECG recordings
    func fetchECGRecordings(limit: Int = 10, startDate: Date? = nil) async throws -> [HKElectrocardiogram] {
        guard isHealthKitAvailable else {
            throw HealthKitError.notAvailable
        }

        var predicate: NSPredicate?
        if let startDate = startDate {
            predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date(), options: .strictStartDate)
        }

        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)

        return try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: ecgType,
                predicate: predicate,
                limit: limit,
                sortDescriptors: [sortDescriptor]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                let ecgs = (samples as? [HKElectrocardiogram]) ?? []
                continuation.resume(returning: ecgs)
            }

            healthStore.execute(query)
        }
    }

    /// Read voltage measurements from an ECG recording
    /// - Parameter ecg: The ECG recording to read
    /// - Returns: Array of voltage measurements in microvolts
    func readECGVoltages(from ecg: HKElectrocardiogram) async throws -> [Double] {
        guard isHealthKitAvailable else {
            throw HealthKitError.notAvailable
        }

        return try await withCheckedThrowingContinuation { continuation in
            var voltages: [Double] = []

            let query = HKElectrocardiogramQuery(ecg) { query, result in
                switch result {
                case .measurement(let measurement):
                    if let voltage = measurement.quantity(for: .appleWatchSimilarToLeadI)?.doubleValue(for: .voltUnit(with: .micro)) {
                        voltages.append(voltage)
                    }

                case .done:
                    continuation.resume(returning: voltages)

                case .error(let error):
                    continuation.resume(throwing: error)

                @unknown default:
                    break
                }
            }

            healthStore.execute(query)
        }
    }

    /// Read ECG data and return processed signal ready for screening
    /// - Parameter ecg: The ECG recording to process
    /// - Returns: Tuple of (processed signals for each 10s strip, sample rate)
    func processECGForScreening(ecg: HKElectrocardiogram) async throws -> ([[Float]], Float) {
        // Read raw voltages
        let voltages = try await readECGVoltages(from: ecg)

        guard !voltages.isEmpty else {
            throw HealthKitError.noData
        }

        // Calculate sample rate (should be ~512 Hz for Apple Watch)
        let sampleRate = Double(voltages.count) / ecg.duration.doubleValue(for: .second())

        // Preprocess and split into 10-second strips
        let processedStrips = ECGPreprocessor.shared.preprocessHealthKitECG(
            voltages: voltages,
            sampleRate: sampleRate
        )

        return (processedStrips, Float(sampleRate))
    }

    // MARK: - Heart Rate Data

    /// Fetch recent heart rate samples
    /// - Parameters:
    ///   - limit: Maximum number of samples
    ///   - startDate: Start date for query
    /// - Returns: Array of heart rate samples (bpm)
    func fetchHeartRateSamples(limit: Int = 100, startDate: Date? = nil) async throws -> [(date: Date, bpm: Double)] {
        guard isHealthKitAvailable else {
            throw HealthKitError.notAvailable
        }

        let startDate = startDate ?? Calendar.current.date(byAdding: .day, value: -7, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date(), options: .strictStartDate)
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)

        return try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: heartRateType,
                predicate: predicate,
                limit: limit,
                sortDescriptors: [sortDescriptor]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                let results = (samples as? [HKQuantitySample])?.map { sample in
                    let bpm = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                    return (date: sample.startDate, bpm: bpm)
                } ?? []

                continuation.resume(returning: results)
            }

            healthStore.execute(query)
        }
    }

    /// Get the latest heart rate reading
    func fetchLatestHeartRate() async throws -> Double? {
        let samples = try await fetchHeartRateSamples(limit: 1)
        let latest = samples.first?.bpm

        await MainActor.run {
            self.latestHeartRate = latest
        }

        return latest
    }

    /// Calculate heart rate statistics for a time period
    func calculateHeartRateStats(days: Int = 7) async throws -> HeartRateStats {
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        let samples = try await fetchHeartRateSamples(limit: 1000, startDate: startDate)

        guard !samples.isEmpty else {
            return HeartRateStats(min: 0, max: 0, average: 0, resting: nil)
        }

        let bpms = samples.map { $0.bpm }
        let min = bpms.min() ?? 0
        let max = bpms.max() ?? 0
        let average = bpms.reduce(0, +) / Double(bpms.count)

        // Estimate resting heart rate (lowest 10% of readings)
        let sorted = bpms.sorted()
        let restingCount = max(1, sorted.count / 10)
        let resting = sorted.prefix(restingCount).reduce(0, +) / Double(restingCount)

        return HeartRateStats(min: min, max: max, average: average, resting: resting)
    }

    // MARK: - Observation

    /// Start observing new ECG recordings
    func startObservingECGRecordings() {
        guard isHealthKitAvailable else { return }

        let query = HKObserverQuery(sampleType: ecgType, predicate: nil) { [weak self] _, completionHandler, error in
            if error == nil {
                Task {
                    if let ecgs = try? await self?.fetchECGRecordings(limit: 5) {
                        await MainActor.run {
                            self?.ecgRecordings = ecgs
                        }
                    }
                }
            }
            completionHandler()
        }

        healthStore.execute(query)
    }
}

// MARK: - Supporting Types

struct HeartRateStats {
    let min: Double
    let max: Double
    let average: Double
    let resting: Double?

    var formattedAverage: String {
        String(format: "%.0f", average)
    }

    var formattedRange: String {
        String(format: "%.0f - %.0f", min, max)
    }

    var formattedResting: String? {
        guard let resting = resting else { return nil }
        return String(format: "%.0f", resting)
    }
}

enum HealthKitError: LocalizedError {
    case notAvailable
    case notAuthorized
    case noData
    case queryFailed(Error)

    var errorDescription: String? {
        switch self {
        case .notAvailable:
            return String(localized: "healthkit.error.notAvailable")
        case .notAuthorized:
            return String(localized: "healthkit.error.notAuthorized")
        case .noData:
            return String(localized: "healthkit.error.noData")
        case .queryFailed(let error):
            return String(localized: "healthkit.error.queryFailed") + ": \(error.localizedDescription)"
        }
    }
}

// MARK: - ECG Recording Extension

extension HKElectrocardiogram {
    /// Formatted date string
    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: startDate)
    }

    /// Duration in seconds
    var durationInSeconds: Double {
        duration.doubleValue(for: .second())
    }

    /// Classification description
    var classificationDescription: String {
        switch classification {
        case .sinusRhythm:
            return String(localized: "ecg.classification.sinusRhythm")
        case .atrialFibrillation:
            return String(localized: "ecg.classification.atrialFibrillation")
        case .inconclusiveLowHeartRate:
            return String(localized: "ecg.classification.lowHeartRate")
        case .inconclusiveHighHeartRate:
            return String(localized: "ecg.classification.highHeartRate")
        case .inconclusivePoorReading:
            return String(localized: "ecg.classification.poorReading")
        case .inconclusiveOther:
            return String(localized: "ecg.classification.inconclusive")
        case .notSet:
            return String(localized: "ecg.classification.notSet")
        @unknown default:
            return String(localized: "ecg.classification.unknown")
        }
    }
}
