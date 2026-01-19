// ECGImageExtractor.swift
// Extract ECG Waveforms from PDF and Image Files
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import UIKit
import PDFKit
import CoreImage
import Accelerate

/// Service for extracting ECG waveforms from PDF and image files
final class ECGImageExtractor {

    // MARK: - Singleton

    static let shared = ECGImageExtractor()

    // MARK: - Configuration

    private let pdfDPI: CGFloat = 300.0
    private let minStripHeight: Int = 20
    private let gapTolerance: Int = 50

    // Color detection mode
    enum WaveformColor {
        case red
        case black
        case blue
        case green
        case auto  // Try to detect automatically
    }

    // Red color detection thresholds
    private let redThreshold: (minR: UInt8, maxG: UInt8, maxB: UInt8) = (150, 180, 180)
    private let hsvRedThreshold: (hue1: ClosedRange<Float>, hue2: ClosedRange<Float>, satMin: Float, valMin: Float) = (
        0...0.055, 0.833...1.0, 0.196, 0.588  // 0-20° and 300-360° hue
    )

    // Black/dark color detection (for printed ECGs)
    private let blackThreshold: (maxR: UInt8, maxG: UInt8, maxB: UInt8, maxSum: Int) = (80, 80, 80, 180)

    // Blue color detection
    private let blueThreshold: (maxR: UInt8, maxG: UInt8, minB: UInt8) = (150, 150, 150)

    // Green color detection
    private let greenThreshold: (maxR: UInt8, minG: UInt8, maxB: UInt8) = (150, 150, 150)

    // MARK: - Initialization

    private init() {}

    // MARK: - Public Methods

    /// Extract ECG signals from a PDF file
    /// - Parameter url: URL to the PDF file
    /// - Returns: Array of extracted ECG signals (one per strip)
    func extractFromPDF(at url: URL) async throws -> ExtractedECGData {
        guard let document = PDFDocument(url: url) else {
            throw ExtractionError.invalidPDF
        }

        guard document.pageCount > 0, let page = document.page(at: 0) else {
            throw ExtractionError.noPages
        }

        // Render PDF to image at high DPI
        let pageRect = page.bounds(for: .mediaBox)
        let scale = pdfDPI / 72.0  // PDF points to pixels

        let width = Int(pageRect.width * scale)
        let height = Int(pageRect.height * scale)

        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        guard let context = UIGraphicsGetCurrentContext() else {
            UIGraphicsEndImageContext()
            throw ExtractionError.renderFailed
        }

        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: scale, y: -scale)

        page.draw(with: .mediaBox, to: context)

        guard let image = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            throw ExtractionError.renderFailed
        }
        UIGraphicsEndImageContext()

        return try await extractFromImage(image)
    }

    /// Extract ECG signals from an image
    /// - Parameter image: UIImage containing ECG traces
    /// - Returns: Extracted ECG data with signals for each strip
    func extractFromImage(_ image: UIImage) async throws -> ExtractedECGData {
        guard let cgImage = image.cgImage else {
            throw ExtractionError.invalidImage
        }

        let width = cgImage.width
        let height = cgImage.height

        // Get pixel data
        guard let pixelData = getPixelData(from: cgImage) else {
            throw ExtractionError.pixelDataFailed
        }

        // Auto-detect waveform color and create mask
        let (waveformMask, detectedColor) = createWaveformMask(pixels: pixelData, width: width, height: height)
        print("[ECGImageExtractor] Detected waveform color: \(detectedColor)")

        // Find ECG strips using horizontal projection
        let strips = findStrips(mask: waveformMask, width: width, height: height)

        guard !strips.isEmpty else {
            throw ExtractionError.noStripsFound
        }

        // Extract waveform from each strip
        var signals: [[Float]] = []
        var sampleRate: Float = 0

        for strip in strips {
            let (signal, rate) = extractWaveformFromStrip(
                mask: waveformMask,
                width: width,
                stripTop: strip.top,
                stripBottom: strip.bottom
            )
            signals.append(signal)
            if rate > 0 {
                sampleRate = rate
            }
        }

        // Estimate sample rate based on image width and typical ECG duration (10 seconds per strip)
        if sampleRate == 0 {
            sampleRate = Float(signals.first?.count ?? 3000) / 10.0
        }

        return ExtractedECGData(
            signals: signals,
            sampleRate: sampleRate,
            stripCount: strips.count,
            imageWidth: width,
            imageHeight: height
        )
    }

    // MARK: - Private Methods

    /// Get raw pixel data from CGImage
    private func getPixelData(from image: CGImage) -> [UInt8]? {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow

        var pixelData = [UInt8](repeating: 0, count: totalBytes)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return pixelData
    }

    /// Create binary mask for waveform pixels (auto-detects color)
    private func createWaveformMask(pixels: [UInt8], width: Int, height: Int) -> (mask: [Bool], color: WaveformColor) {
        // Count pixels for each color type to auto-detect
        var redCount = 0
        var blackCount = 0
        var blueCount = 0
        var greenCount = 0

        // Sample pixels to detect dominant waveform color
        let sampleStep = max(1, (width * height) / 100000)  // Sample up to 100k pixels
        for i in stride(from: 0, to: width * height, by: sampleStep) {
            let offset = i * 4
            let r = pixels[offset]
            let g = pixels[offset + 1]
            let b = pixels[offset + 2]

            if isRed(r: r, g: g, b: b) { redCount += 1 }
            if isBlack(r: r, g: g, b: b) { blackCount += 1 }
            if isBlue(r: r, g: g, b: b) { blueCount += 1 }
            if isGreen(r: r, g: g, b: b) { greenCount += 1 }
        }

        // Determine dominant color (prefer red for Apple Watch PDFs)
        let detectedColor: WaveformColor
        let maxCount = max(redCount, blackCount, blueCount, greenCount)

        if maxCount == 0 {
            detectedColor = .black  // Default to black if nothing detected
        } else if redCount == maxCount || (redCount > 0 && redCount >= maxCount / 2) {
            detectedColor = .red  // Prefer red if it's significant
        } else if blackCount == maxCount {
            detectedColor = .black
        } else if blueCount == maxCount {
            detectedColor = .blue
        } else {
            detectedColor = .green
        }

        print("[ECGImageExtractor] Color counts - Red: \(redCount), Black: \(blackCount), Blue: \(blueCount), Green: \(greenCount)")

        // Create mask using detected color
        var mask = [Bool](repeating: false, count: width * height)

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let r = pixels[offset]
                let g = pixels[offset + 1]
                let b = pixels[offset + 2]

                let isWaveform: Bool
                switch detectedColor {
                case .red:
                    isWaveform = isRed(r: r, g: g, b: b)
                case .black:
                    isWaveform = isBlack(r: r, g: g, b: b)
                case .blue:
                    isWaveform = isBlue(r: r, g: g, b: b)
                case .green:
                    isWaveform = isGreen(r: r, g: g, b: b)
                case .auto:
                    // Try all colors
                    isWaveform = isRed(r: r, g: g, b: b) || isBlack(r: r, g: g, b: b) ||
                                 isBlue(r: r, g: g, b: b) || isGreen(r: r, g: g, b: b)
                }

                mask[y * width + x] = isWaveform
            }
        }

        return (mask, detectedColor)
    }

    /// Check if pixel is red
    private func isRed(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        // RGB threshold check
        let rgbMatch = r >= redThreshold.minR &&
                       g <= redThreshold.maxG &&
                       b <= redThreshold.maxB &&
                       r > g && r > b  // Red must be dominant

        // HSV check for better red detection
        let hsvMatch = isRedInHSV(r: r, g: g, b: b)

        return rgbMatch || hsvMatch
    }

    /// Check if pixel is black/dark
    private func isBlack(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        let sum = Int(r) + Int(g) + Int(b)
        return r <= blackThreshold.maxR &&
               g <= blackThreshold.maxG &&
               b <= blackThreshold.maxB &&
               sum <= blackThreshold.maxSum
    }

    /// Check if pixel is blue
    private func isBlue(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        return r <= blueThreshold.maxR &&
               g <= blueThreshold.maxG &&
               b >= blueThreshold.minB &&
               b > r && b > g  // Blue must be dominant
    }

    /// Check if pixel is green
    private func isGreen(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        return r <= greenThreshold.maxR &&
               g >= greenThreshold.minG &&
               b <= greenThreshold.maxB &&
               g > r && g > b  // Green must be dominant
    }

    /// Check if RGB color is red in HSV space
    private func isRedInHSV(r: UInt8, g: UInt8, b: UInt8) -> Bool {
        let rf = Float(r) / 255.0
        let gf = Float(g) / 255.0
        let bf = Float(b) / 255.0

        let maxC = max(rf, gf, bf)
        let minC = min(rf, gf, bf)
        let delta = maxC - minC

        // Value
        let v = maxC

        // Saturation
        let s = maxC > 0 ? delta / maxC : 0

        // Hue
        var h: Float = 0
        if delta > 0 {
            if maxC == rf {
                h = (gf - bf) / delta
                if h < 0 { h += 6 }
            } else if maxC == gf {
                h = 2 + (bf - rf) / delta
            } else {
                h = 4 + (rf - gf) / delta
            }
            h /= 6
        }

        // Check red hue ranges and saturation/value thresholds
        let isRedHue = hsvRedThreshold.hue1.contains(h) || hsvRedThreshold.hue2.contains(h)
        return isRedHue && s >= hsvRedThreshold.satMin && v >= hsvRedThreshold.valMin
    }

    /// Find ECG strips using horizontal projection
    private func findStrips(mask: [Bool], width: Int, height: Int) -> [(top: Int, bottom: Int)] {
        // Calculate horizontal projection (sum of red pixels per row)
        var projection = [Int](repeating: 0, count: height)
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] {
                    projection[y] += 1
                }
            }
        }

        // Find peaks in projection (strip locations)
        var strips: [(top: Int, bottom: Int)] = []
        var inStrip = false
        var stripStart = 0

        let threshold = width / 50  // Minimum pixels to be considered part of a strip

        for y in 0..<height {
            if projection[y] > threshold {
                if !inStrip {
                    inStrip = true
                    stripStart = y
                }
            } else {
                if inStrip {
                    let stripHeight = y - stripStart
                    if stripHeight >= minStripHeight {
                        strips.append((top: stripStart, bottom: y))
                    }
                    inStrip = false
                }
            }
        }

        // Handle case where last strip extends to bottom
        if inStrip && height - stripStart >= minStripHeight {
            strips.append((top: stripStart, bottom: height))
        }

        return strips
    }

    /// Extract waveform values from a single strip
    private func extractWaveformFromStrip(
        mask: [Bool],
        width: Int,
        stripTop: Int,
        stripBottom: Int
    ) -> (signal: [Float], sampleRate: Float) {
        let stripHeight = stripBottom - stripTop
        var signal = [Float](repeating: 0, count: width)
        var validColumns = 0

        for x in 0..<width {
            // Find red pixels in this column within the strip
            var redYPositions: [Int] = []

            for y in stripTop..<stripBottom {
                if mask[y * width + x] {
                    redYPositions.append(y - stripTop)
                }
            }

            if redYPositions.isEmpty {
                // No signal in this column - will be interpolated
                signal[x] = Float.nan
            } else {
                // Use median of red pixel positions (skeleton approach)
                redYPositions.sort()
                let median = redYPositions[redYPositions.count / 2]

                // Normalize to [-1, 1] range (inverted because image Y is top-down)
                let normalized = 1.0 - 2.0 * Float(median) / Float(stripHeight)
                signal[x] = normalized
                validColumns += 1
            }
        }

        // Interpolate gaps
        signal = interpolateGaps(signal, maxGap: gapTolerance)

        // Remove any remaining NaN values
        signal = signal.map { $0.isNaN ? 0 : $0 }

        // Estimate sample rate (assuming 10 seconds per strip)
        let sampleRate = Float(validColumns) / 10.0

        return (signal, sampleRate)
    }

    /// Interpolate small gaps in the signal
    private func interpolateGaps(_ signal: [Float], maxGap: Int) -> [Float] {
        var result = signal
        var gapStart: Int?

        for i in 0..<signal.count {
            if signal[i].isNaN {
                if gapStart == nil {
                    gapStart = i
                }
            } else {
                if let start = gapStart {
                    let gapLength = i - start
                    if gapLength <= maxGap && start > 0 {
                        // Interpolate
                        let startValue = signal[start - 1]
                        let endValue = signal[i]
                        for j in start..<i {
                            let t = Float(j - start + 1) / Float(gapLength + 1)
                            result[j] = startValue + t * (endValue - startValue)
                        }
                    }
                    gapStart = nil
                }
            }
        }

        return result
    }
}

// MARK: - Supporting Types

struct ExtractedECGData {
    let signals: [[Float]]
    let sampleRate: Float
    let stripCount: Int
    let imageWidth: Int
    let imageHeight: Int

    /// Get preprocessed signals ready for model inference
    func getProcessedSignals() -> [[Float]] {
        return signals.map { signal in
            ECGPreprocessor.shared.preprocess(signal: signal, sourceSampleRate: sampleRate)
        }
    }

    /// Combine all strips into a single continuous signal
    func getCombinedSignal() -> [Float] {
        return signals.flatMap { $0 }
    }
}

enum ExtractionError: LocalizedError {
    case invalidPDF
    case noPages
    case renderFailed
    case invalidImage
    case pixelDataFailed
    case noStripsFound
    case processingFailed

    var errorDescription: String? {
        switch self {
        case .invalidPDF:
            return String(localized: "extraction.error.invalidPDF")
        case .noPages:
            return String(localized: "extraction.error.noPages")
        case .renderFailed:
            return String(localized: "extraction.error.renderFailed")
        case .invalidImage:
            return String(localized: "extraction.error.invalidImage")
        case .pixelDataFailed:
            return String(localized: "extraction.error.pixelDataFailed")
        case .noStripsFound:
            return String(localized: "extraction.error.noStripsFound")
        case .processingFailed:
            return String(localized: "extraction.error.processingFailed")
        }
    }
}

// MARK: - Camera/Photo Processing

extension ECGImageExtractor {

    /// Process an image from camera or photo library
    /// - Parameter image: The captured/selected image
    /// - Returns: Extracted ECG data
    func processUserImage(_ image: UIImage) async throws -> ExtractedECGData {
        // Optionally resize very large images to improve processing speed
        let maxDimension: CGFloat = 4000
        let processedImage: UIImage

        if max(image.size.width, image.size.height) > maxDimension {
            let scale = maxDimension / max(image.size.width, image.size.height)
            let newSize = CGSize(width: image.size.width * scale, height: image.size.height * scale)

            UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
            image.draw(in: CGRect(origin: .zero, size: newSize))
            processedImage = UIGraphicsGetImageFromCurrentImageContext() ?? image
            UIGraphicsEndImageContext()
        } else {
            processedImage = image
        }

        return try await extractFromImage(processedImage)
    }
}
