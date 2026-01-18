// PDFReportGenerator.swift
// PDF Report Generation for ECG Screening Results
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import UIKit
import PDFKit

/// Service for generating PDF reports from screening results
final class PDFReportGenerator {

    // MARK: - Singleton

    static let shared = PDFReportGenerator()

    // MARK: - Configuration

    private let pageSize = CGSize(width: 612, height: 792)  // US Letter
    private let margin: CGFloat = 50
    private let headerHeight: CGFloat = 100
    private let footerHeight: CGFloat = 80

    // Colors
    private let primaryColor = UIColor(red: 0.8, green: 0.1, blue: 0.1, alpha: 1.0)  // Medical red
    private let textColor = UIColor.black
    private let secondaryTextColor = UIColor.darkGray
    private let backgroundColor = UIColor.white

    // MARK: - Initialization

    private init() {}

    // MARK: - Public Methods

    /// Generate a PDF report for a screening result
    /// - Parameters:
    ///   - result: The screening result to report
    ///   - patientInfo: Optional patient information
    /// - Returns: PDF data
    func generateReport(
        for result: ScreeningResultEntity,
        patientInfo: PatientInfo? = nil
    ) -> Data {
        let renderer = UIGraphicsPDFRenderer(bounds: CGRect(origin: .zero, size: pageSize))

        return renderer.pdfData { context in
            context.beginPage()

            var yPosition: CGFloat = margin

            // Header
            yPosition = drawHeader(in: context.cgContext, at: yPosition)

            // Patient Info Section
            if let info = patientInfo {
                yPosition = drawPatientInfo(info, in: context.cgContext, at: yPosition)
            }

            // Screening Date
            yPosition = drawScreeningDate(result.createdAt, in: context.cgContext, at: yPosition)

            // Primary Finding
            yPosition = drawPrimaryFinding(result, in: context.cgContext, at: yPosition)

            // Top Conditions Table
            yPosition = drawConditionsTable(result, in: context.cgContext, at: yPosition)

            // ECG Waveform (if signal data available)
            if let ecgRecord = result.ecgRecord, let signal = ecgRecord.signal {
                yPosition = drawECGWaveform(signal, in: context.cgContext, at: yPosition)
            }

            // Disclaimer
            drawDisclaimer(in: context.cgContext, at: pageSize.height - footerHeight - margin)

            // Footer
            drawFooter(in: context.cgContext)
        }
    }

    /// Generate and save report to a temporary file
    /// - Parameters:
    ///   - result: The screening result
    ///   - patientInfo: Optional patient information
    /// - Returns: URL to the generated PDF file
    func generateReportFile(
        for result: ScreeningResultEntity,
        patientInfo: PatientInfo? = nil
    ) throws -> URL {
        let pdfData = generateReport(for: result, patientInfo: patientInfo)

        let fileName = "ECG_Report_\(formatDateForFilename(result.createdAt)).pdf"
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        try pdfData.write(to: fileURL)

        return fileURL
    }

    // MARK: - Drawing Methods

    private func drawHeader(in context: CGContext, at yPosition: CGFloat) -> CGFloat {
        let contentWidth = pageSize.width - 2 * margin

        // Company Logo/Name
        let titleFont = UIFont.systemFont(ofSize: 24, weight: .bold)
        let title = Constants.App.displayName
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: primaryColor
        ]

        let titleRect = CGRect(x: margin, y: yPosition, width: contentWidth, height: 30)
        title.draw(in: titleRect, withAttributes: titleAttributes)

        // Subtitle
        let subtitleFont = UIFont.systemFont(ofSize: 12, weight: .regular)
        let subtitle = String(localized: "report.subtitle")
        let subtitleAttributes: [NSAttributedString.Key: Any] = [
            .font: subtitleFont,
            .foregroundColor: secondaryTextColor
        ]

        let subtitleRect = CGRect(x: margin, y: yPosition + 35, width: contentWidth, height: 20)
        subtitle.draw(in: subtitleRect, withAttributes: subtitleAttributes)

        // Horizontal line
        context.setStrokeColor(primaryColor.cgColor)
        context.setLineWidth(2)
        context.move(to: CGPoint(x: margin, y: yPosition + 60))
        context.addLine(to: CGPoint(x: pageSize.width - margin, y: yPosition + 60))
        context.strokePath()

        return yPosition + 80
    }

    private func drawPatientInfo(_ info: PatientInfo, in context: CGContext, at yPosition: CGFloat) -> CGFloat {
        let sectionFont = UIFont.systemFont(ofSize: 14, weight: .semibold)
        let contentFont = UIFont.systemFont(ofSize: 12, weight: .regular)
        let contentWidth = pageSize.width - 2 * margin

        // Section Title
        let sectionTitle = String(localized: "report.patientInfo")
        let sectionAttributes: [NSAttributedString.Key: Any] = [
            .font: sectionFont,
            .foregroundColor: textColor
        ]
        sectionTitle.draw(at: CGPoint(x: margin, y: yPosition), withAttributes: sectionAttributes)

        var currentY = yPosition + 25

        // Patient details in two columns
        let leftColumnX = margin
        let rightColumnX = margin + contentWidth / 2
        let lineHeight: CGFloat = 18

        let contentAttributes: [NSAttributedString.Key: Any] = [
            .font: contentFont,
            .foregroundColor: textColor
        ]

        // Left column
        if let name = info.name {
            "\(String(localized: "report.name")): \(name)".draw(
                at: CGPoint(x: leftColumnX, y: currentY),
                withAttributes: contentAttributes
            )
        }

        if let dob = info.dateOfBirth {
            let dobString = formatDate(dob)
            "\(String(localized: "report.dob")): \(dobString)".draw(
                at: CGPoint(x: leftColumnX, y: currentY + lineHeight),
                withAttributes: contentAttributes
            )
        }

        // Right column
        if let gender = info.gender {
            "\(String(localized: "report.gender")): \(gender)".draw(
                at: CGPoint(x: rightColumnX, y: currentY),
                withAttributes: contentAttributes
            )
        }

        if let id = info.patientID {
            "\(String(localized: "report.id")): \(id)".draw(
                at: CGPoint(x: rightColumnX, y: currentY + lineHeight),
                withAttributes: contentAttributes
            )
        }

        return currentY + lineHeight * 2 + 20
    }

    private func drawScreeningDate(_ date: Date, in context: CGContext, at yPosition: CGFloat) -> CGFloat {
        let font = UIFont.systemFont(ofSize: 12, weight: .regular)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: secondaryTextColor
        ]

        let dateString = "\(String(localized: "report.screeningDate")): \(formatDateTime(date))"
        dateString.draw(at: CGPoint(x: margin, y: yPosition), withAttributes: attributes)

        return yPosition + 30
    }

    private func drawPrimaryFinding(_ result: ScreeningResultEntity, in context: CGContext, at yPosition: CGFloat) -> CGFloat {
        let contentWidth = pageSize.width - 2 * margin

        // Section box
        let boxRect = CGRect(x: margin, y: yPosition, width: contentWidth, height: 60)
        context.setFillColor(UIColor(white: 0.95, alpha: 1.0).cgColor)
        context.fill(boxRect)
        context.setStrokeColor(primaryColor.cgColor)
        context.setLineWidth(1)
        context.stroke(boxRect)

        // Primary finding title
        let titleFont = UIFont.systemFont(ofSize: 12, weight: .semibold)
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: secondaryTextColor
        ]
        String(localized: "report.primaryFinding").draw(
            at: CGPoint(x: margin + 15, y: yPosition + 10),
            withAttributes: titleAttributes
        )

        // Condition name
        let conditionFont = UIFont.systemFont(ofSize: 18, weight: .bold)
        let conditionName = result.primaryCondition?.localizedName ?? "Unknown"
        let conditionAttributes: [NSAttributedString.Key: Any] = [
            .font: conditionFont,
            .foregroundColor: textColor
        ]
        conditionName.draw(
            at: CGPoint(x: margin + 15, y: yPosition + 30),
            withAttributes: conditionAttributes
        )

        // Confidence
        let confidenceFont = UIFont.systemFont(ofSize: 14, weight: .medium)
        let confidenceText = String(format: "%.1f%%", result.primaryConfidence * 100)
        let confidenceAttributes: [NSAttributedString.Key: Any] = [
            .font: confidenceFont,
            .foregroundColor: primaryColor
        ]
        confidenceText.draw(
            at: CGPoint(x: pageSize.width - margin - 60, y: yPosition + 32),
            withAttributes: confidenceAttributes
        )

        return yPosition + 80
    }

    private func drawConditionsTable(_ result: ScreeningResultEntity, in context: CGContext, at yPosition: CGFloat) -> CGFloat {
        let contentWidth = pageSize.width - 2 * margin

        // Section Title
        let titleFont = UIFont.systemFont(ofSize: 14, weight: .semibold)
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: textColor
        ]
        String(localized: "report.topFindings").draw(
            at: CGPoint(x: margin, y: yPosition),
            withAttributes: titleAttributes
        )

        var currentY = yPosition + 25

        // Table header
        let headerFont = UIFont.systemFont(ofSize: 10, weight: .semibold)
        let headerAttributes: [NSAttributedString.Key: Any] = [
            .font: headerFont,
            .foregroundColor: secondaryTextColor
        ]

        String(localized: "report.condition").draw(
            at: CGPoint(x: margin, y: currentY),
            withAttributes: headerAttributes
        )
        String(localized: "report.probability").draw(
            at: CGPoint(x: pageSize.width - margin - 80, y: currentY),
            withAttributes: headerAttributes
        )

        currentY += 15

        // Divider
        context.setStrokeColor(UIColor.lightGray.cgColor)
        context.setLineWidth(0.5)
        context.move(to: CGPoint(x: margin, y: currentY))
        context.addLine(to: CGPoint(x: pageSize.width - margin, y: currentY))
        context.strokePath()

        currentY += 5

        // Table rows
        let rowFont = UIFont.systemFont(ofSize: 11, weight: .regular)
        let topConditions = result.topConditions(count: 10, threshold: 0.05)

        for (condition, probability) in topConditions {
            let rowAttributes: [NSAttributedString.Key: Any] = [
                .font: rowFont,
                .foregroundColor: textColor
            ]

            condition.localizedName.draw(
                at: CGPoint(x: margin, y: currentY),
                withAttributes: rowAttributes
            )

            let probText = String(format: "%.1f%%", probability * 100)
            let probAttributes: [NSAttributedString.Key: Any] = [
                .font: rowFont,
                .foregroundColor: probability > 0.5 ? primaryColor : secondaryTextColor
            ]
            probText.draw(
                at: CGPoint(x: pageSize.width - margin - 50, y: currentY),
                withAttributes: probAttributes
            )

            currentY += 18
        }

        return currentY + 20
    }

    private func drawECGWaveform(_ signal: [Float], in context: CGContext, at yPosition: CGFloat) -> CGFloat {
        let contentWidth = pageSize.width - 2 * margin
        let waveformHeight: CGFloat = 100

        // Section Title
        let titleFont = UIFont.systemFont(ofSize: 14, weight: .semibold)
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: textColor
        ]
        String(localized: "report.ecgWaveform").draw(
            at: CGPoint(x: margin, y: yPosition),
            withAttributes: titleAttributes
        )

        let waveformY = yPosition + 25

        // Draw waveform box
        let boxRect = CGRect(x: margin, y: waveformY, width: contentWidth, height: waveformHeight)
        context.setStrokeColor(UIColor.lightGray.cgColor)
        context.setLineWidth(0.5)
        context.stroke(boxRect)

        // Draw waveform
        guard !signal.isEmpty else { return waveformY + waveformHeight + 20 }

        // Downsample if needed for display
        let displayPoints = min(signal.count, Int(contentWidth))
        let step = signal.count / displayPoints

        context.setStrokeColor(primaryColor.cgColor)
        context.setLineWidth(1)

        let midY = waveformY + waveformHeight / 2
        let amplitude = waveformHeight / 2 - 10

        context.move(to: CGPoint(x: margin, y: midY))

        for i in 0..<displayPoints {
            let sampleIndex = min(i * step, signal.count - 1)
            let value = CGFloat(signal[sampleIndex])
            let x = margin + CGFloat(i) / CGFloat(displayPoints) * contentWidth
            let y = midY - value * amplitude

            context.addLine(to: CGPoint(x: x, y: y))
        }

        context.strokePath()

        return waveformY + waveformHeight + 20
    }

    private func drawDisclaimer(in context: CGContext, at yPosition: CGFloat) {
        let contentWidth = pageSize.width - 2 * margin

        // Warning box
        let boxRect = CGRect(x: margin, y: yPosition, width: contentWidth, height: 60)
        context.setFillColor(UIColor(red: 1.0, green: 0.95, blue: 0.9, alpha: 1.0).cgColor)
        context.fill(boxRect)
        context.setStrokeColor(UIColor.orange.cgColor)
        context.setLineWidth(1)
        context.stroke(boxRect)

        // Disclaimer text
        let titleFont = UIFont.systemFont(ofSize: 10, weight: .bold)
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: UIColor.orange
        ]
        String(localized: "disclaimer.title").draw(
            at: CGPoint(x: margin + 10, y: yPosition + 8),
            withAttributes: titleAttributes
        )

        let textFont = UIFont.systemFont(ofSize: 8, weight: .regular)
        let textAttributes: [NSAttributedString.Key: Any] = [
            .font: textFont,
            .foregroundColor: secondaryTextColor
        ]

        let disclaimerText = String(localized: "disclaimer.full")
        let textRect = CGRect(x: margin + 10, y: yPosition + 22, width: contentWidth - 20, height: 35)
        disclaimerText.draw(in: textRect, withAttributes: textAttributes)
    }

    private func drawFooter(in context: CGContext) {
        let footerY = pageSize.height - margin

        // Copyright
        let font = UIFont.systemFont(ofSize: 8, weight: .regular)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: secondaryTextColor
        ]

        let copyright = "© \(Constants.App.copyrightYear) \(Constants.App.companyName). All rights reserved."
        copyright.draw(at: CGPoint(x: margin, y: footerY - 10), withAttributes: attributes)

        // Model version
        let versionText = "Model: \(Constants.App.modelVersion)"
        let versionWidth = versionText.size(withAttributes: attributes).width
        versionText.draw(
            at: CGPoint(x: pageSize.width - margin - versionWidth, y: footerY - 10),
            withAttributes: attributes
        )
    }

    // MARK: - Helpers

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }

    private func formatDateTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    private func formatDateForFilename(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter.string(from: date)
    }
}

// MARK: - Patient Info

struct PatientInfo {
    var name: String?
    var dateOfBirth: Date?
    var gender: String?
    var patientID: String?

    init(name: String? = nil, dateOfBirth: Date? = nil, gender: String? = nil, patientID: String? = nil) {
        self.name = name
        self.dateOfBirth = dateOfBirth
        self.gender = gender
        self.patientID = patientID
    }

    /// Create from user profile
    static func from(profile: UserProfileEntity) -> PatientInfo {
        PatientInfo(
            dateOfBirth: profile.dateOfBirth,
            gender: profile.gender
        )
    }
}
