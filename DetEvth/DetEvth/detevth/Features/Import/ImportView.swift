import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ImportView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var viewModel = ImportViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Text("import.source.title")
                    .font(.headline)
                    .padding(.top)

                // Import Options
                VStack(spacing: 12) {
                    ImportOptionButton(
                        title: String(localized: "import.source.pdf"),
                        subtitle: String(localized: "import.source.pdf.desc"),
                        icon: "doc.fill",
                        color: .red
                    ) {
                        viewModel.showDocumentPicker = true
                    }

                    ImportOptionButton(
                        title: String(localized: "import.source.photos"),
                        subtitle: String(localized: "import.source.photos.desc"),
                        icon: "photo.fill",
                        color: .green
                    ) {
                        viewModel.showPhotoPicker = true
                    }
                }
                .padding(.horizontal)

                Divider()
                    .padding(.horizontal)

                // Tips Section
                VStack(alignment: .leading, spacing: 12) {
                    Text("import.formats")
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    VStack(alignment: .leading, spacing: 4) {
                        Label(String(localized: "import.format.pdf"), systemImage: "doc.fill")
                        Label(String(localized: "import.format.image"), systemImage: "photo")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    Text("import.tips")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .padding(.top, 8)

                    VStack(alignment: .leading, spacing: 4) {
                        Label(String(localized: "import.tip1"), systemImage: "checkmark.circle")
                        Label(String(localized: "import.tip2"), systemImage: "checkmark.circle")
                        Label(String(localized: "import.tip3"), systemImage: "checkmark.circle")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)

                Spacer()
            }
            .navigationTitle("import.title")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("common.cancel") { dismiss() }
                }
            }
            .sheet(isPresented: $viewModel.showDocumentPicker) {
                DocumentPicker(onDocumentPicked: { url in
                    viewModel.processPDF(url: url)
                })
            }
            .photosPicker(
                isPresented: $viewModel.showPhotoPicker,
                selection: $viewModel.selectedPhoto,
                matching: .images
            )
            .onChange(of: viewModel.selectedPhoto) { _, newValue in
                if newValue != nil {
                    viewModel.processSelectedPhoto()
                }
            }
            .overlay {
                if viewModel.isProcessing {
                    ProcessingOverlay(message: viewModel.processingMessage)
                }
            }
            .alert("import.error.title", isPresented: $viewModel.showError) {
                Button("common.ok", role: .cancel) {}
            } message: {
                Text(viewModel.errorMessage)
            }
            .navigationDestination(isPresented: $viewModel.showResults) {
                if let result = viewModel.screeningResult {
                    ResultDetailView(screening: result)
                }
            }
        }
    }
}

struct ImportOptionButton: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(color)
                    .frame(width: 40)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
        }
        .buttonStyle(.plain)
    }
}

struct ProcessingOverlay: View {
    let message: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.5)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .tint(.white)

                Text(message)
                    .font(.headline)
                    .foregroundStyle(.white)
            }
            .padding(40)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }
}

@MainActor
class ImportViewModel: ObservableObject {
    @Published var showDocumentPicker = false
    @Published var showPhotoPicker = false
    @Published var selectedPhoto: PhotosPickerItem?
    @Published var isProcessing = false
    @Published var processingMessage = ""
    @Published var showError = false
    @Published var errorMessage = ""
    @Published var showResults = false
    @Published var screeningResult: ScreeningResultModel?

    private let extractor = ECGImageExtractor.shared
    private let inferenceService = ECGInferenceService.shared
    private let repository = ScreeningRepository.shared

    func processPDF(url: URL) {
        isProcessing = true
        processingMessage = String(localized: "import.processing.extracting")

        Task {
            do {
                // Start accessing security-scoped resource
                guard url.startAccessingSecurityScopedResource() else {
                    throw ImportError.accessDenied
                }
                defer { url.stopAccessingSecurityScopedResource() }

                // Extract ECG data from PDF
                let extractedData = try await extractor.extractFromPDF(at: url)

                await MainActor.run {
                    processingMessage = String(localized: "import.processing.analyzing")
                }

                // Run inference
                let inferenceResult = try await inferenceService.runScreeningMultiStrip(
                    signals: extractedData.getProcessedSignals()
                )

                // Save to Core Data
                let combinedSignal = extractedData.getCombinedSignal()
                let ecgRecord = repository.saveECGRecord(
                    source: .importPDF,
                    signal: combinedSignal,
                    sampleRate: extractedData.sampleRate,
                    duration: Float(combinedSignal.count) / extractedData.sampleRate,
                    fileName: url.lastPathComponent
                )

                let savedResult = repository.saveScreeningResult(
                    for: ecgRecord,
                    probabilities: inferenceResult.probabilities
                )

                // Extract heart rate from signal
                let heartRate = ECGHeartRateExtractor.extractHeartRate(
                    from: combinedSignal,
                    sampleRate: extractedData.sampleRate
                )

                await MainActor.run {
                    isProcessing = false
                    // Create result model for navigation
                    self.screeningResult = ScreeningResultModel(
                        id: savedResult.id ?? UUID(),
                        date: savedResult.createdAt ?? Date(),
                        primaryCondition: savedResult.primaryCondition?.localizedName ?? String(localized: "common.unknown"),
                        confidence: savedResult.primaryConfidence,
                        source: "import",
                        signal: combinedSignal,
                        probabilities: inferenceResult.probabilities,
                        heartRate: heartRate
                    )
                    showResults = true
                }

            } catch {
                await MainActor.run {
                    isProcessing = false
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }

    func processSelectedPhoto() {
        guard let selectedPhoto = selectedPhoto else { return }

        isProcessing = true
        processingMessage = String(localized: "import.processing.loading")

        Task {
            do {
                // Load image from PhotosPickerItem
                guard let data = try await selectedPhoto.loadTransferable(type: Data.self),
                      let image = UIImage(data: data) else {
                    throw ImportError.invalidImage
                }

                await MainActor.run {
                    processingMessage = String(localized: "import.processing.extracting")
                }

                // Extract ECG data from image
                let extractedData = try await extractor.processUserImage(image)

                await MainActor.run {
                    processingMessage = String(localized: "import.processing.analyzing")
                }

                // Run inference
                let inferenceResult = try await inferenceService.runScreeningMultiStrip(
                    signals: extractedData.getProcessedSignals()
                )

                // Save to Core Data
                let combinedSignal = extractedData.getCombinedSignal()
                let ecgRecord = repository.saveECGRecord(
                    source: .importImage,
                    signal: combinedSignal,
                    sampleRate: extractedData.sampleRate,
                    duration: Float(combinedSignal.count) / extractedData.sampleRate,
                    fileName: nil
                )

                let savedResult = repository.saveScreeningResult(
                    for: ecgRecord,
                    probabilities: inferenceResult.probabilities
                )

                // Extract heart rate from signal
                let heartRate = ECGHeartRateExtractor.extractHeartRate(
                    from: combinedSignal,
                    sampleRate: extractedData.sampleRate
                )

                await MainActor.run {
                    isProcessing = false
                    self.selectedPhoto = nil
                    // Create result model for navigation
                    self.screeningResult = ScreeningResultModel(
                        id: savedResult.id ?? UUID(),
                        date: savedResult.createdAt ?? Date(),
                        primaryCondition: savedResult.primaryCondition?.localizedName ?? String(localized: "common.unknown"),
                        confidence: savedResult.primaryConfidence,
                        source: "import",
                        signal: combinedSignal,
                        probabilities: inferenceResult.probabilities,
                        heartRate: heartRate
                    )
                    showResults = true
                }

            } catch {
                await MainActor.run {
                    isProcessing = false
                    self.selectedPhoto = nil
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
}

enum ImportError: LocalizedError {
    case accessDenied
    case invalidImage

    var errorDescription: String? {
        switch self {
        case .accessDenied:
            return String(localized: "import.error.accessDenied")
        case .invalidImage:
            return String(localized: "import.error.invalidImage")
        }
    }
}

// MARK: - Document Picker

struct DocumentPicker: UIViewControllerRepresentable {
    let onDocumentPicked: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.pdf])
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onDocumentPicked: onDocumentPicked)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onDocumentPicked: (URL) -> Void

        init(onDocumentPicked: @escaping (URL) -> Void) {
            self.onDocumentPicked = onDocumentPicked
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onDocumentPicked(url)
        }
    }
}

#Preview {
    ImportView()
}
