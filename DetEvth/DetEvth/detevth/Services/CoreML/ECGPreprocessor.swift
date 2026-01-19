// ECGPreprocessor.swift
// ECG Signal Preprocessing for ECGFounder Model
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import Accelerate

/// Preprocessor for ECG signals to match ECGFounder model requirements
///
/// Pipeline:
/// 1. Resample to 500 Hz
/// 2. Bandpass filter (0.67 - 40 Hz, Butterworth N=4)
/// 3. Notch filter (50 Hz, Q=30)
/// 4. Median baseline removal
/// 5. Z-score normalization
final class ECGPreprocessor {

    // MARK: - Singleton

    static let shared = ECGPreprocessor()

    // MARK: - Initialization

    private init() {}

    // MARK: - Main Preprocessing

    /// Preprocess raw ECG signal for model inference
    /// - Parameters:
    ///   - signal: Raw ECG signal
    ///   - sourceSampleRate: Original sample rate of the signal (e.g., 512 Hz from HealthKit)
    /// - Returns: Preprocessed signal ready for inference (5000 samples)
    func preprocess(signal: [Float], sourceSampleRate: Float) -> [Float] {
        // Step 1: Resample to target sample rate (500 Hz)
        var processed = resample(signal, from: sourceSampleRate, to: Constants.ECG.targetSampleRate)

        // Step 2: Bandpass filter (0.67 - 40 Hz)
        processed = bandpassFilter(
            processed,
            sampleRate: Constants.ECG.targetSampleRate,
            lowCutoff: Constants.ECG.bandpassLowCutoff,
            highCutoff: Constants.ECG.bandpassHighCutoff
        )

        // Step 3: Notch filter (50 Hz) - only if sample rate allows
        if Constants.ECG.targetSampleRate > 100 {
            processed = notchFilter(
                processed,
                sampleRate: Constants.ECG.targetSampleRate,
                frequency: Constants.ECG.notchFrequency
            )
        }

        // Step 4: Median baseline removal
        processed = removeBaseline(processed, sampleRate: Constants.ECG.targetSampleRate)

        // Step 5: Z-score normalization
        processed = zScoreNormalize(processed)

        // Step 6: Ensure correct length
        processed = adjustLength(processed, targetLength: Constants.ECG.targetLength)

        return processed
    }

    // MARK: - Resampling

    /// Resample signal from source rate to target rate
    private func resample(_ signal: [Float], from sourceRate: Float, to targetRate: Float) -> [Float] {
        guard sourceRate != targetRate else { return signal }

        let targetLength = Int(Float(signal.count) * targetRate / sourceRate)
        var resampled = [Float](repeating: 0, count: targetLength)

        // Use vDSP for efficient resampling with linear interpolation
        for i in 0..<targetLength {
            let sourceIndex = Float(i) * Float(signal.count - 1) / Float(targetLength - 1)
            let lowerIndex = Int(sourceIndex)
            let upperIndex = min(lowerIndex + 1, signal.count - 1)
            let fraction = sourceIndex - Float(lowerIndex)

            resampled[i] = signal[lowerIndex] * (1 - fraction) + signal[upperIndex] * fraction
        }

        return resampled
    }

    // MARK: - Bandpass Filter

    /// Apply Butterworth bandpass filter
    /// Approximates scipy.signal.butter with N=4, btype='bandpass'
    private func bandpassFilter(_ signal: [Float], sampleRate: Float, lowCutoff: Float, highCutoff: Float) -> [Float] {
        // Apply high-pass filter first (removes DC offset and low frequency)
        var filtered = highpassFilter(signal, sampleRate: sampleRate, cutoff: lowCutoff)

        // Then apply low-pass filter
        filtered = lowpassFilter(filtered, sampleRate: sampleRate, cutoff: highCutoff)

        return filtered
    }

    /// Simple high-pass filter using moving average subtraction
    private func highpassFilter(_ signal: [Float], sampleRate: Float, cutoff: Float) -> [Float] {
        // Calculate window size for moving average (approximates high-pass)
        let windowSize = max(3, Int(sampleRate / cutoff / 2))
        let halfWindow = windowSize / 2

        var filtered = [Float](repeating: 0, count: signal.count)
        var runningSum: Float = 0

        // Calculate initial sum
        for i in 0..<min(windowSize, signal.count) {
            runningSum += signal[i]
        }

        // Apply moving average high-pass
        for i in 0..<signal.count {
            let avgStart = max(0, i - halfWindow)
            let avgEnd = min(signal.count - 1, i + halfWindow)
            let windowLength = avgEnd - avgStart + 1

            var windowSum: Float = 0
            for j in avgStart...avgEnd {
                windowSum += signal[j]
            }

            let baseline = windowSum / Float(windowLength)
            filtered[i] = signal[i] - baseline
        }

        return filtered
    }

    /// Simple low-pass filter using exponential moving average
    private func lowpassFilter(_ signal: [Float], sampleRate: Float, cutoff: Float) -> [Float] {
        // Calculate smoothing factor (alpha) based on cutoff frequency
        let dt: Float = 1.0 / sampleRate
        let rc: Float = 1.0 / (2.0 * Float.pi * cutoff)
        let alpha: Float = dt / (rc + dt)

        var filtered = [Float](repeating: 0, count: signal.count)
        filtered[0] = signal[0]

        // Apply exponential moving average (forward pass)
        for i in 1..<signal.count {
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i - 1]
        }

        // Apply backward pass for zero-phase filtering
        var backward = filtered
        for i in (0..<(signal.count - 1)).reversed() {
            backward[i] = alpha * backward[i] + (1 - alpha) * backward[i + 1]
        }

        return backward
    }

    // MARK: - Notch Filter

    /// Apply notch filter to remove power-line interference
    private func notchFilter(_ signal: [Float], sampleRate: Float, frequency: Float) -> [Float] {
        // Simple notch filter implementation
        // Approximates scipy.signal.iirnotch

        let Q: Float = 30.0  // Quality factor
        let w0 = 2.0 * Float.pi * frequency / sampleRate

        // Calculate filter coefficients
        let alpha = sin(w0) / (2.0 * Q)
        let b0: Float = 1.0
        let b1: Float = -2.0 * cos(w0)
        let b2: Float = 1.0
        let a0: Float = 1.0 + alpha
        let a1: Float = -2.0 * cos(w0)
        let a2: Float = 1.0 - alpha

        // Normalize coefficients
        let nb0 = b0 / a0
        let nb1 = b1 / a0
        let nb2 = b2 / a0
        let na1 = a1 / a0
        let na2 = a2 / a0

        // Apply biquad filter (forward pass)
        var filtered = [Float](repeating: 0, count: signal.count)
        var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0

        for i in 0..<signal.count {
            let x0 = signal[i]
            let y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2

            filtered[i] = y0

            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0
        }

        // Backward pass for zero-phase filtering
        x1 = 0; x2 = 0; y1 = 0; y2 = 0
        var backward = [Float](repeating: 0, count: signal.count)

        for i in (0..<signal.count).reversed() {
            let x0 = filtered[i]
            let y0 = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2

            backward[i] = y0

            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0
        }

        return backward
    }

    // MARK: - Baseline Removal

    /// Remove baseline wander using median filter
    private func removeBaseline(_ signal: [Float], sampleRate: Float) -> [Float] {
        // Calculate kernel size (0.4 seconds)
        var kernelSize = Int(Constants.ECG.baselineKernelDuration * sampleRate)
        if kernelSize % 2 == 0 {
            kernelSize += 1  // Ensure odd kernel size
        }

        let halfKernel = kernelSize / 2
        var baseline = [Float](repeating: 0, count: signal.count)

        // Apply median filter for baseline estimation
        for i in 0..<signal.count {
            let start = max(0, i - halfKernel)
            let end = min(signal.count - 1, i + halfKernel)

            var window = Array(signal[start...end])
            window.sort()

            baseline[i] = window[window.count / 2]
        }

        // Subtract baseline
        var result = [Float](repeating: 0, count: signal.count)
        vDSP_vsub(baseline, 1, signal, 1, &result, 1, vDSP_Length(signal.count))

        return result
    }

    // MARK: - Z-Score Normalization

    /// Apply z-score normalization
    private func zScoreNormalize(_ signal: [Float]) -> [Float] {
        var mean: Float = 0
        var std: Float = 0

        // Calculate mean
        vDSP_meanv(signal, 1, &mean, vDSP_Length(signal.count))

        // Calculate standard deviation
        var sumOfSquaredDiffs: Float = 0
        var temp = [Float](repeating: 0, count: signal.count)

        // Subtract mean
        var negativeMean = -mean
        vDSP_vsadd(signal, 1, &negativeMean, &temp, 1, vDSP_Length(signal.count))

        // Square differences
        vDSP_vsq(temp, 1, &temp, 1, vDSP_Length(signal.count))

        // Sum
        vDSP_sve(temp, 1, &sumOfSquaredDiffs, vDSP_Length(signal.count))

        std = sqrt(sumOfSquaredDiffs / Float(signal.count))

        // Avoid division by zero
        if std < 1e-8 {
            std = 1.0
        }

        // Normalize: (x - mean) / std
        var normalized = [Float](repeating: 0, count: signal.count)
        vDSP_vsadd(signal, 1, &negativeMean, &normalized, 1, vDSP_Length(signal.count))

        var invStd = 1.0 / std
        vDSP_vsmul(normalized, 1, &invStd, &normalized, 1, vDSP_Length(signal.count))

        return normalized
    }

    // MARK: - Length Adjustment

    /// Adjust signal to target length (pad or truncate)
    private func adjustLength(_ signal: [Float], targetLength: Int) -> [Float] {
        if signal.count == targetLength {
            return signal
        } else if signal.count < targetLength {
            // Pad with zeros
            var padded = signal
            padded.append(contentsOf: [Float](repeating: 0, count: targetLength - signal.count))
            return padded
        } else {
            // Truncate
            return Array(signal.prefix(targetLength))
        }
    }
}

// MARK: - Extension for HealthKit ECG Data

extension ECGPreprocessor {

    /// Preprocess HealthKit ECG data
    /// - Parameters:
    ///   - voltages: Array of voltage measurements from HealthKit (microvolts)
    ///   - sampleRate: Sample rate from HealthKit (typically 512 Hz)
    /// - Returns: Preprocessed signals for each 10-second strip
    func preprocessHealthKitECG(voltages: [Double], sampleRate: Double) -> [[Float]] {
        // Convert to Float and from microvolts to millivolts
        let signalInMV = voltages.map { Float($0 / 1000.0) }

        // HealthKit provides 30 seconds at 512 Hz = 15360 samples
        // We need to split into three 10-second strips

        let samplesPerStrip = Int(sampleRate * 10)
        var strips: [[Float]] = []

        for i in 0..<3 {
            let start = i * samplesPerStrip
            let end = min(start + samplesPerStrip, signalInMV.count)

            if start < signalInMV.count {
                let strip = Array(signalInMV[start..<end])
                let processed = preprocess(signal: strip, sourceSampleRate: Float(sampleRate))
                strips.append(processed)
            }
        }

        return strips
    }
}
