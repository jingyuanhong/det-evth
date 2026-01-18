// ScreeningService.swift
// ECG Disease Screening Orchestration Service
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import Combine
import HealthKit

/// Service that orchestrates the complete ECG screening workflow
final class ScreeningService: ObservableObject {

    // MARK: - Singleton

    static let shared = ScreeningService()

    // MARK: - Published Properties

    @Published var isProcessing = false
    @Published var progress: Float = 0
    @Published var statusMessage = ""
    @Published var currentResult: ScreeningResultEntity?
    @Published var error: Error?

    // MARK: - Dependencies

    private let healthKitService = HealthKitService.shared
    private let imageExtractor = ECGImageExtractor.shared
    private let preprocessor = ECGPreprocessor.shared
    private let inferenceService = ECGInferenceService.shared
    private let repository = ScreeningRepository.shared

    // MARK: - Initialization

    private init() {}

    // MARK: - Screening Methods

    /// Run screening on a HealthKit ECG recording
    /// - Parameter ecg: The HealthKit ECG sample
    /// - Returns: The screening result
    @MainActor
    func screenHealthKitECG(_ ecg: HKElectrocardiogram) async throws -> ScreeningResultEntity {
        reset()
        isProcessing = true

        do {
            // Step 1: Read voltages
            updateProgress(0.1, message: String(localized: "screening.reading"))
            let (processedStrips, sampleRate) = try await healthKitService.processECGForScreening(ecg: ecg)

            // Step 2: Save ECG record
            updateProgress(0.3, message: String(localized: "screening.processing"))
            let combinedSignal = processedStrips.flatMap { $0 }
            let ecgRecord = repository.saveECGRecord(
                source: .healthkit,
                signal: combinedSignal,
                sampleRate: sampleRate,
                duration: Float(ecg.durationInSeconds)
            )

            // Step 3: Run inference
            updateProgress(0.5, message: String(localized: "screening.analyzing"))
            let result = try await inferenceService.runScreeningMultiStrip(signals: processedStrips)

            // Step 4: Save results
            updateProgress(0.9, message: String(localized: "screening.saving"))
            let screeningResult = repository.saveScreeningResult(
                for: ecgRecord,
                probabilities: result.probabilities,
                perStripProbabilities: processedStrips.isEmpty ? nil : [result.probabilities]
            )

            updateProgress(1.0, message: String(localized: "screening.complete"))
            currentResult = screeningResult
            isProcessing = false

            return screeningResult

        } catch {
            self.error = error
            isProcessing = false
            throw error
        }
    }

    /// Run screening on a PDF file
    /// - Parameter url: URL to the PDF file
    /// - Returns: The screening result
    @MainActor
    func screenPDF(at url: URL) async throws -> ScreeningResultEntity {
        reset()
        isProcessing = true

        do {
            // Step 1: Extract ECG from PDF
            updateProgress(0.1, message: String(localized: "screening.extracting"))
            let extractedData = try await imageExtractor.extractFromPDF(at: url)

            // Step 2: Preprocess signals
            updateProgress(0.3, message: String(localized: "screening.processing"))
            let processedStrips = extractedData.getProcessedSignals()

            // Step 3: Save ECG record
            let combinedSignal = processedStrips.flatMap { $0 }
            let ecgRecord = repository.saveECGRecord(
                source: .importPDF,
                signal: combinedSignal,
                sampleRate: Constants.ECG.targetSampleRate,
                duration: Float(extractedData.stripCount) * Float(Constants.ECG.duration),
                fileName: url.lastPathComponent
            )

            // Step 4: Run inference
            updateProgress(0.6, message: String(localized: "screening.analyzing"))
            let result = try await inferenceService.runScreeningMultiStrip(signals: processedStrips)

            // Step 5: Save results
            updateProgress(0.9, message: String(localized: "screening.saving"))
            let screeningResult = repository.saveScreeningResult(
                for: ecgRecord,
                probabilities: result.probabilities
            )

            updateProgress(1.0, message: String(localized: "screening.complete"))
            currentResult = screeningResult
            isProcessing = false

            return screeningResult

        } catch {
            self.error = error
            isProcessing = false
            throw error
        }
    }

    /// Run screening on an image (from camera or photo library)
    /// - Parameter image: The UIImage containing ECG traces
    /// - Returns: The screening result
    @MainActor
    func screenImage(_ image: UIImage, source: ECGSource = .importImage) async throws -> ScreeningResultEntity {
        reset()
        isProcessing = true

        do {
            // Step 1: Extract ECG from image
            updateProgress(0.1, message: String(localized: "screening.extracting"))
            let extractedData = try await imageExtractor.processUserImage(image)

            // Step 2: Preprocess signals
            updateProgress(0.3, message: String(localized: "screening.processing"))
            let processedStrips = extractedData.getProcessedSignals()

            guard !processedStrips.isEmpty else {
                throw ScreeningError.noSignalExtracted
            }

            // Step 3: Save ECG record
            let combinedSignal = processedStrips.flatMap { $0 }
            let ecgRecord = repository.saveECGRecord(
                source: source,
                signal: combinedSignal,
                sampleRate: Constants.ECG.targetSampleRate,
                duration: Float(extractedData.stripCount) * Float(Constants.ECG.duration)
            )

            // Step 4: Run inference
            updateProgress(0.6, message: String(localized: "screening.analyzing"))
            let result = try await inferenceService.runScreeningMultiStrip(signals: processedStrips)

            // Step 5: Save results
            updateProgress(0.9, message: String(localized: "screening.saving"))
            let screeningResult = repository.saveScreeningResult(
                for: ecgRecord,
                probabilities: result.probabilities
            )

            updateProgress(1.0, message: String(localized: "screening.complete"))
            currentResult = screeningResult
            isProcessing = false

            return screeningResult

        } catch {
            self.error = error
            isProcessing = false
            throw error
        }
    }

    /// Run screening on raw ECG signal data
    /// - Parameters:
    ///   - signal: Raw ECG signal
    ///   - sampleRate: Sample rate of the signal
    ///   - source: Source of the ECG data
    /// - Returns: The screening result
    @MainActor
    func screenRawSignal(_ signal: [Float], sampleRate: Float, source: ECGSource = .appleWatch) async throws -> ScreeningResultEntity {
        reset()
        isProcessing = true

        do {
            // Step 1: Preprocess
            updateProgress(0.2, message: String(localized: "screening.processing"))
            let processed = preprocessor.preprocess(signal: signal, sourceSampleRate: sampleRate)

            // Step 2: Save ECG record
            updateProgress(0.4, message: String(localized: "screening.saving"))
            let ecgRecord = repository.saveECGRecord(
                source: source,
                signal: processed,
                sampleRate: Constants.ECG.targetSampleRate,
                duration: Float(signal.count) / sampleRate
            )

            // Step 3: Run inference
            updateProgress(0.6, message: String(localized: "screening.analyzing"))
            let result = try await inferenceService.runScreening(signal: processed)

            // Step 4: Save results
            updateProgress(0.9, message: String(localized: "screening.saving"))
            let screeningResult = repository.saveScreeningResult(
                for: ecgRecord,
                probabilities: result.probabilities
            )

            updateProgress(1.0, message: String(localized: "screening.complete"))
            currentResult = screeningResult
            isProcessing = false

            return screeningResult

        } catch {
            self.error = error
            isProcessing = false
            throw error
        }
    }

    // MARK: - Private Methods

    private func reset() {
        progress = 0
        statusMessage = ""
        error = nil
        currentResult = nil
    }

    private func updateProgress(_ value: Float, message: String) {
        progress = value
        statusMessage = message
    }
}

// MARK: - Screening Error

enum ScreeningError: LocalizedError {
    case noSignalExtracted
    case preprocessingFailed
    case inferenceFailed
    case saveFailed

    var errorDescription: String? {
        switch self {
        case .noSignalExtracted:
            return String(localized: "screening.error.noSignal")
        case .preprocessingFailed:
            return String(localized: "screening.error.preprocessing")
        case .inferenceFailed:
            return String(localized: "screening.error.inference")
        case .saveFailed:
            return String(localized: "screening.error.save")
        }
    }
}

// MARK: - Screening View Model

@MainActor
final class ScreeningViewModel: ObservableObject {

    @Published var isProcessing = false
    @Published var progress: Float = 0
    @Published var statusMessage = ""
    @Published var result: ScreeningResultEntity?
    @Published var error: Error?
    @Published var showError = false

    private let screeningService = ScreeningService.shared
    private var cancellables = Set<AnyCancellable>()

    init() {
        // Observe screening service
        screeningService.$isProcessing
            .receive(on: DispatchQueue.main)
            .assign(to: &$isProcessing)

        screeningService.$progress
            .receive(on: DispatchQueue.main)
            .assign(to: &$progress)

        screeningService.$statusMessage
            .receive(on: DispatchQueue.main)
            .assign(to: &$statusMessage)

        screeningService.$currentResult
            .receive(on: DispatchQueue.main)
            .assign(to: &$result)

        screeningService.$error
            .receive(on: DispatchQueue.main)
            .sink { [weak self] error in
                self?.error = error
                self?.showError = error != nil
            }
            .store(in: &cancellables)
    }

    func screenHealthKitECG(_ ecg: HKElectrocardiogram) async {
        do {
            _ = try await screeningService.screenHealthKitECG(ecg)
        } catch {
            // Error is handled via published property
        }
    }

    func screenPDF(at url: URL) async {
        do {
            _ = try await screeningService.screenPDF(at: url)
        } catch {
            // Error is handled via published property
        }
    }

    func screenImage(_ image: UIImage, source: ECGSource) async {
        do {
            _ = try await screeningService.screenImage(image, source: source)
        } catch {
            // Error is handled via published property
        }
    }
}
