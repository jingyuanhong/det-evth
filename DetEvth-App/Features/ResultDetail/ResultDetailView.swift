import SwiftUI
import Charts

struct ResultDetailView: View {
    let screening: ScreeningResultModel

    @State private var showShareSheet = false
    @State private var showReportPreview = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Primary Finding Card
                PrimaryFindingCard(
                    condition: screening.primaryCondition,
                    confidence: screening.confidence
                )

                // ECG Waveform
                WaveformSection()

                // Top Conditions
                TopConditionsSection()

                // Disclaimer
                DisclaimerBanner()

                // Action Buttons
                HStack(spacing: 12) {
                    Button {
                        showReportPreview = true
                    } label: {
                        Label(String(localized: "results.generateReport"), systemImage: "doc.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        showShareSheet = true
                    } label: {
                        Label(String(localized: "results.share"), systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(.horizontal)

                Spacer(minLength: 50)
            }
            .padding(.top)
        }
        .navigationTitle("results.title")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showReportPreview) {
            ReportPreviewView()
        }
        .sheet(isPresented: $showShareSheet) {
            // Share sheet placeholder
            Text("Share Sheet")
        }
    }
}

// MARK: - Primary Finding Card
struct PrimaryFindingCard: View {
    let condition: String
    let confidence: Float

    var body: some View {
        VStack(spacing: 12) {
            Text("results.primary")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Image(systemName: confidence > 0.8 ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                    .foregroundStyle(confidence > 0.8 ? .green : .orange)
                    .font(.title)

                Text(condition)
                    .font(.title2)
                    .fontWeight(.semibold)
            }

            Text(String(localized: "results.confidence", defaultValue: "Confidence: \(String(format: "%.1f", confidence * 100))%"))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
        .padding(.horizontal)
    }
}

// MARK: - Waveform Section
struct WaveformSection: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("results.waveform")
                .font(.headline)
                .padding(.horizontal)

            // Placeholder waveform chart
            Chart {
                ForEach(0..<100, id: \.self) { index in
                    LineMark(
                        x: .value("Time", index),
                        y: .value("Voltage", sin(Double(index) * 0.2) + Double.random(in: -0.1...0.1))
                    )
                    .foregroundStyle(.red)
                }
            }
            .frame(height: 150)
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
            .padding(.horizontal)
        }
    }
}

// MARK: - Top Conditions Section
struct TopConditionsSection: View {
    // Sample conditions for preview
    let conditions = [
        ("Normal Sinus Rhythm", 0.92),
        ("Sinus Rhythm", 0.89),
        ("Sinus Tachycardia", 0.35),
        ("Left Axis Deviation", 0.23),
        ("T Wave Abnormality", 0.18)
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("results.topConditions")
                .font(.headline)
                .padding(.horizontal)

            VStack(spacing: 8) {
                ForEach(conditions, id: \.0) { condition, probability in
                    ConditionRow(name: condition, probability: Float(probability))
                }
            }
            .padding(.horizontal)
        }
    }
}

struct ConditionRow: View {
    let name: String
    let probability: Float

    var body: some View {
        HStack {
            Text(name)
                .font(.subheadline)

            Spacer()

            Text("\(String(format: "%.1f", probability * 100))%")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(probability > 0.5 ? .primary : .secondary)
        }
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Disclaimer Banner
struct DisclaimerBanner: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(String(localized: "results.disclaimer.title"), systemImage: "exclamationmark.triangle.fill")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.orange)

            Text("results.disclaimer.text")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
        .padding(.horizontal)
    }
}

// MARK: - Report Preview
struct ReportPreviewView: View {
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            Text("PDF Report Preview")
                .navigationTitle("results.generateReport")
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("common.done") { dismiss() }
                    }
                }
        }
    }
}

#Preview {
    NavigationStack {
        ResultDetailView(screening: ScreeningResultModel(
            id: UUID(),
            date: Date(),
            primaryCondition: "Normal Sinus Rhythm",
            confidence: 0.92,
            source: "healthkit"
        ))
    }
}
