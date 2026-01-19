// ECGFounderModelHelper.swift
// Auto-generated helper for ECGFounder CoreML model
// © 2026 minuscule health Ltd. All rights reserved.

import CoreML
import Foundation

/// Configuration for ECG signal preprocessing
enum ECGConfig {
    static let sampleRate: Float = 500.0
    static let duration: Float = 10.0
    static let signalLength: Int = 5000
    static let numClasses: Int = 150
}

/// Helper class for ECGFounder model inference
class ECGFounderHelper {

    private let model: MLModel

    init() throws {
        // Load the quantized model (4-bit)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        guard let modelURL = Bundle.main.url(
            forResource: "ECGFounder1Lead_4bit",
            withExtension: "mlmodelc"
        ) else {
            throw ECGFounderError.modelNotFound
        }

        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }

    /// Run inference on preprocessed ECG signal
    /// - Parameter signal: Preprocessed ECG signal (5000 samples, z-score normalized)
    /// - Returns: Array of 150 probabilities for each disease class
    func predict(signal: [Float]) throws -> [Float] {
        guard signal.count == ECGConfig.signalLength else {
            throw ECGFounderError.invalidInputLength(expected: ECGConfig.signalLength, got: signal.count)
        }

        // Create MLMultiArray input (1, 1, 5000)
        let inputShape = [1, 1, ECGConfig.signalLength] as [NSNumber]
        let inputArray = try MLMultiArray(shape: inputShape, dataType: .float32)

        for i in 0..<signal.count {
            inputArray[i] = NSNumber(value: signal[i])
        }

        // Create feature provider
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "ecg_signal": MLFeatureValue(multiArray: inputArray)
        ])

        // Run prediction
        let output = try model.prediction(from: inputFeatures)

        // Extract logits and apply sigmoid
        guard let logitsArray = output.featureValue(for: "disease_logits")?.multiArrayValue else {
            throw ECGFounderError.outputExtractionFailed
        }

        var probabilities = [Float](repeating: 0, count: ECGConfig.numClasses)
        for i in 0..<ECGConfig.numClasses {
            let logit = logitsArray[i].floatValue
            probabilities[i] = 1.0 / (1.0 + exp(-logit))  // sigmoid
        }

        return probabilities
    }
}

enum ECGFounderError: Error {
    case modelNotFound
    case invalidInputLength(expected: Int, got: Int)
    case outputExtractionFailed
}
