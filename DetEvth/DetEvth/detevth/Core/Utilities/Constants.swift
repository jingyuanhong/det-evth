// Constants.swift
// Application Constants for DetEvth
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation

enum Constants {
    // MARK: - App Info
    enum App {
        static let name = "det-evth"
        static let displayName = "DetEvth"
        static let companyName = "minuscule health Ltd"
        static let companyLocation = "London, United Kingdom"
        static let copyrightYear = "2026"
        static let version = "1.0.0"
        static let modelVersion = "ECGFounder-1L-v1.0"
    }

    // Legacy accessors for compatibility
    static let appName = App.name
    static let companyName = App.companyName
    static let companyLocation = App.companyLocation
    static let copyrightYear = App.copyrightYear

    // MARK: - ECG Processing
    enum ECG {
        static let targetSampleRate: Float = 500.0
        static let targetLength: Int = 5000
        static let duration: TimeInterval = 10.0
        static let numClasses: Int = 150

        // HealthKit ECG sample rate
        static let healthKitSampleRate: Float = 512.0
        static let healthKitDuration: TimeInterval = 30.0  // Apple Watch records 30 seconds

        // Filter parameters (matching ECGFounder preprocessing)
        static let bandpassLowCutoff: Float = 0.67
        static let bandpassHighCutoff: Float = 40.0
        static let notchFrequency: Float = 50.0
        static let notchQ: Float = 30.0
        static let baselineKernelDuration: Float = 0.4  // seconds
        static let baselineKernelFactor: Float = 0.4  // Deprecated, use baselineKernelDuration
    }

    // MARK: - Model
    enum Model {
        static let name = "ECGFounder1Lead"
        static let inputShape = (1, 1, 5000)
        static let numClasses = 150
        static let probabilityThreshold: Float = 0.1
    }

    // MARK: - Image Extraction
    enum ImageExtraction {
        static let pdfDPI: CGFloat = 300.0
        static let minStripHeight: Int = 30
        static let gapTolerance: Int = 30

        // BGR thresholds (relaxed for JPG)
        static let bgrBlueMax: UInt8 = 180
        static let bgrGreenMax: UInt8 = 180
        static let bgrRedMin: UInt8 = 150

        // HSV thresholds
        static let hsvHueRed1: ClosedRange<UInt8> = 0...10
        static let hsvHueRed2: ClosedRange<UInt8> = 150...180
        static let hsvSaturationMin: UInt8 = 50
        static let hsvValueMin: UInt8 = 150
    }

    // MARK: - UI
    enum UI {
        static let primaryColor = "AccentColor"
        static let animationDuration: Double = 0.3
        static let cornerRadius: CGFloat = 12.0
    }

    // MARK: - Storage Keys
    enum StorageKeys {
        static let selectedLanguage = "selectedLanguage"
        static let hasCompletedOnboarding = "hasCompletedOnboarding"
        static let notificationsEnabled = "notificationsEnabled"
    }
}
