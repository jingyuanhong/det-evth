// ECGInferenceService.swift
// ECG Disease Screening using CoreML
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import CoreML
import Accelerate

/// Service for running ECG disease screening inference
final class ECGInferenceService {

    // MARK: - Singleton

    static let shared = ECGInferenceService()

    // MARK: - Properties

    private var model: MLModel?
    private let modelQueue = DispatchQueue(label: "com.minusculehealth.ecginference", qos: .userInitiated)

    // MARK: - Initialization

    private init() {
        loadModel()
    }

    // MARK: - Model Loading

    private func loadModel() {
        modelQueue.async { [weak self] in
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine

                // Try to load the quantized model first, fall back to full precision
                if let modelURL = Bundle.main.url(forResource: "ECGFounder1Lead_4bit", withExtension: "mlmodelc") {
                    self?.model = try MLModel(contentsOf: modelURL, configuration: config)
                    print("[ECGInferenceService] Loaded quantized model (4-bit)")
                } else if let modelURL = Bundle.main.url(forResource: "ECGFounder1Lead", withExtension: "mlmodelc") {
                    self?.model = try MLModel(contentsOf: modelURL, configuration: config)
                    print("[ECGInferenceService] Loaded full precision model")
                } else {
                    print("[ECGInferenceService] ERROR: Model not found in bundle")
                }
            } catch {
                print("[ECGInferenceService] ERROR loading model: \(error)")
            }
        }
    }

    // MARK: - Inference

    /// Run disease screening on preprocessed ECG signal
    /// - Parameter signal: Preprocessed ECG signal (5000 samples, z-score normalized)
    /// - Returns: ScreeningResult with disease probabilities
    func runScreening(signal: [Float]) async throws -> ScreeningResult {
        guard let model = model else {
            throw ECGInferenceError.modelNotLoaded
        }

        guard signal.count == Constants.ECG.targetLength else {
            throw ECGInferenceError.invalidInputLength(
                expected: Constants.ECG.targetLength,
                got: signal.count
            )
        }

        return try await withCheckedThrowingContinuation { continuation in
            modelQueue.async {
                do {
                    // Create MLMultiArray input (1, 1, 5000)
                    let inputShape = [1, 1, Constants.ECG.targetLength] as [NSNumber]
                    let inputArray = try MLMultiArray(shape: inputShape, dataType: .float32)

                    // Copy signal data
                    let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: signal.count)
                    signal.withUnsafeBufferPointer { buffer in
                        ptr.update(from: buffer.baseAddress!, count: signal.count)
                    }

                    // Create feature provider
                    let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
                        "ecg_signal": MLFeatureValue(multiArray: inputArray)
                    ])

                    // Run prediction
                    let output = try model.prediction(from: inputFeatures)

                    // Extract logits
                    guard let logitsArray = output.featureValue(for: "disease_logits")?.multiArrayValue else {
                        throw ECGInferenceError.outputExtractionFailed
                    }

                    // Apply sigmoid to get probabilities
                    var probabilities = [Float](repeating: 0, count: Constants.ECG.numClasses)
                    for i in 0..<Constants.ECG.numClasses {
                        let logit = logitsArray[i].floatValue
                        probabilities[i] = Self.sigmoid(logit)
                    }

                    // Create result
                    let result = ScreeningResult(probabilities: probabilities)
                    continuation.resume(returning: result)

                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Run screening on multiple ECG strips and average results
    /// - Parameter signals: Array of preprocessed ECG signals
    /// - Returns: Averaged ScreeningResult
    func runScreeningMultiStrip(signals: [[Float]]) async throws -> ScreeningResult {
        var allProbabilities: [[Float]] = []

        for signal in signals {
            let result = try await runScreening(signal: signal)
            allProbabilities.append(result.probabilities)
        }

        // Average probabilities across strips
        var averagedProbs = [Float](repeating: 0, count: Constants.ECG.numClasses)
        for i in 0..<Constants.ECG.numClasses {
            var sum: Float = 0
            for probs in allProbabilities {
                sum += probs[i]
            }
            averagedProbs[i] = sum / Float(allProbabilities.count)
        }

        return ScreeningResult(probabilities: averagedProbs)
    }

    // MARK: - Helpers

    private static func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }
}

// MARK: - Error Types

enum ECGInferenceError: LocalizedError {
    case modelNotLoaded
    case invalidInputLength(expected: Int, got: Int)
    case outputExtractionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "ECG model not loaded. Please try again."
        case .invalidInputLength(let expected, let got):
            return "Invalid ECG signal length. Expected \(expected), got \(got)."
        case .outputExtractionFailed:
            return "Failed to extract model output."
        }
    }
}

// MARK: - Screening Result

struct ScreeningResult {
    let probabilities: [Float]
    let timestamp: Date

    init(probabilities: [Float], timestamp: Date = Date()) {
        self.probabilities = probabilities
        self.timestamp = timestamp
    }

    /// Get top N conditions by probability
    func topConditions(n: Int = 10, threshold: Float = 0.1) -> [(condition: DiseaseCondition, probability: Float)] {
        let conditions = DiseaseConditions.all

        var results: [(condition: DiseaseCondition, probability: Float)] = []

        for (index, probability) in probabilities.enumerated() {
            if probability >= threshold, index < conditions.count {
                results.append((conditions[index], probability))
            }
        }

        // Sort by probability descending
        results.sort { $0.probability > $1.probability }

        return Array(results.prefix(n))
    }

    /// Get the primary (highest probability) condition
    var primaryCondition: (condition: DiseaseCondition, probability: Float)? {
        guard let maxIndex = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) else {
            return nil
        }

        let conditions = DiseaseConditions.all
        guard maxIndex < conditions.count else { return nil }

        return (conditions[maxIndex], probabilities[maxIndex])
    }

    /// Get conditions by category
    func conditions(for category: DiseaseCondition.Category, threshold: Float = 0.1) -> [(condition: DiseaseCondition, probability: Float)] {
        let conditions = DiseaseConditions.all

        var results: [(condition: DiseaseCondition, probability: Float)] = []

        for (index, probability) in probabilities.enumerated() {
            if probability >= threshold,
               index < conditions.count,
               conditions[index].category == category {
                results.append((conditions[index], probability))
            }
        }

        results.sort { $0.probability > $1.probability }
        return results
    }
}
