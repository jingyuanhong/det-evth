import SwiftUI

struct RecordView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var viewModel = RecordViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 32) {
                Spacer()

                // Apple Watch Icon
                Image(systemName: "applewatch")
                    .font(.system(size: 80))
                    .foregroundStyle(.secondary)

                // Instructions
                Text("record.instruction")
                    .font(.title3)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                // Recording Progress
                if viewModel.isRecording {
                    VStack(spacing: 16) {
                        Text(String(localized: "record.recording", defaultValue: "Recording: \(viewModel.recordingSeconds)s / 30s"))
                            .font(.headline)

                        ProgressView(value: Double(viewModel.recordingSeconds), total: 30)
                            .progressViewStyle(.linear)
                            .padding(.horizontal, 40)

                        // Live Waveform Placeholder
                        ECGWaveformView(data: viewModel.liveWaveform)
                            .frame(height: 100)
                            .padding(.horizontal)
                    }
                }

                Spacer()

                // Tips
                Label(String(localized: "record.tips"), systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                // Action Button
                if viewModel.isRecording {
                    Button(role: .destructive) {
                        viewModel.cancelRecording()
                    } label: {
                        Text("record.cancel")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.horizontal)
                } else {
                    Button {
                        viewModel.startRecording()
                    } label: {
                        Label(String(localized: "home.quickActions.record"), systemImage: "waveform.path.ecg")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .padding(.horizontal)
                }

                Spacer()
            }
            .navigationTitle("record.title")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("common.cancel") { dismiss() }
                }
            }
            .onChange(of: viewModel.isComplete) { _, complete in
                if complete {
                    dismiss()
                }
            }
        }
    }
}

// MARK: - ECG Waveform View
struct ECGWaveformView: View {
    let data: [Float]

    var body: some View {
        GeometryReader { geometry in
            Path { path in
                guard !data.isEmpty else { return }

                let width = geometry.size.width
                let height = geometry.size.height
                let midY = height / 2

                let step = width / CGFloat(data.count - 1)

                path.move(to: CGPoint(x: 0, y: midY - CGFloat(data[0]) * midY * 0.8))

                for (index, value) in data.enumerated() {
                    let x = CGFloat(index) * step
                    let y = midY - CGFloat(value) * midY * 0.8
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            .stroke(Color.red, lineWidth: 1.5)
        }
    }
}

// MARK: - ViewModel
class RecordViewModel: ObservableObject {
    @Published var isRecording = false
    @Published var recordingSeconds = 0
    @Published var liveWaveform: [Float] = []
    @Published var isComplete = false

    private var timer: Timer?

    func startRecording() {
        isRecording = true
        recordingSeconds = 0
        liveWaveform = []

        // Simulate recording with timer
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.recordingSeconds += 1

            // Generate fake waveform data
            self.liveWaveform = (0..<100).map { _ in
                Float.random(in: -0.5...1.0)
            }

            if self.recordingSeconds >= 30 {
                self.completeRecording()
            }
        }
    }

    func cancelRecording() {
        timer?.invalidate()
        timer = nil
        isRecording = false
        recordingSeconds = 0
        liveWaveform = []
    }

    private func completeRecording() {
        timer?.invalidate()
        timer = nil
        isRecording = false

        // TODO: Process the ECG data
        // For now, just mark as complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.isComplete = true
        }
    }
}

#Preview {
    RecordView()
}
