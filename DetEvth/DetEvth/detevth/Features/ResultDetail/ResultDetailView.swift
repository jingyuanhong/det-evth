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
                WaveformSection(signal: screening.signal)

                // Top Conditions
                TopConditionsSection(probabilities: screening.probabilities)

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

            Text(String(format: String(localized: "results.confidence"), confidence * 100))
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
    let signal: [Float]?

    // Downsample signal for display
    private var displaySignal: [(index: Int, value: Double)] {
        guard let signal = signal, !signal.isEmpty else {
            return []
        }

        // Downsample to max 500 points for smooth display
        let maxPoints = 500
        let step = max(1, signal.count / maxPoints)

        var result: [(index: Int, value: Double)] = []
        for i in stride(from: 0, to: signal.count, by: step) {
            result.append((index: result.count, value: Double(signal[i])))
        }
        return result
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("results.waveform")
                .font(.headline)
                .padding(.horizontal)

            if displaySignal.isEmpty {
                // No signal data available
                Text("results.waveform.noData")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                    .frame(height: 150)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(.horizontal)
            } else {
                // Real waveform chart
                Chart {
                    ForEach(displaySignal, id: \.index) { point in
                        LineMark(
                            x: .value("Time", point.index),
                            y: .value("Voltage", point.value)
                        )
                        .foregroundStyle(.red)
                        .lineStyle(StrokeStyle(lineWidth: 1))
                    }
                }
                .frame(height: 150)
                .chartXAxis(.hidden)
                .chartYAxis(.hidden)
                .padding(.horizontal)
            }
        }
    }
}

// MARK: - Top Conditions Section
struct TopConditionsSection: View {
    let probabilities: [Float]?

    // Get top conditions from probabilities
    private var topConditions: [(name: String, probability: Float)] {
        guard let probs = probabilities, !probs.isEmpty else {
            return []
        }

        let conditions = DiseaseConditions.all

        // Create array of (index, probability) and sort by probability
        var indexed: [(index: Int, prob: Float)] = []
        for (index, prob) in probs.enumerated() {
            if index < conditions.count && prob >= 0.05 {  // Only show conditions with >= 5% probability
                indexed.append((index, prob))
            }
        }

        indexed.sort { $0.prob > $1.prob }

        // Take top 10
        return indexed.prefix(10).map { item in
            (conditions[item.index].nameEN, item.prob)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("results.topConditions")
                .font(.headline)
                .padding(.horizontal)

            if topConditions.isEmpty {
                Text("results.conditions.noData")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            } else {
                VStack(spacing: 8) {
                    ForEach(topConditions, id: \.name) { condition in
                        ConditionRow(name: condition.name, probability: condition.probability)
                    }
                }
                .padding(.horizontal)
            }
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
                .lineLimit(1)

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
            source: "healthkit",
            signal: (0..<1000).map { Float(sin(Double($0) * 0.05)) },
            probabilities: nil,
            heartRate: 72
        ))
    }
}
