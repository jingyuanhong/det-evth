import SwiftUI
import PhotosUI

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
                        title: String(localized: "import.source.camera"),
                        subtitle: String(localized: "import.source.camera.desc"),
                        icon: "camera.fill",
                        color: .blue
                    ) {
                        viewModel.showCamera = true
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
                // Document Picker
                Text("Document Picker Placeholder")
            }
            .sheet(isPresented: $viewModel.showCamera) {
                // Camera View
                Text("Camera Placeholder")
            }
            .photosPicker(
                isPresented: $viewModel.showPhotoPicker,
                selection: $viewModel.selectedPhoto,
                matching: .images
            )
            .overlay {
                if viewModel.isProcessing {
                    ProcessingOverlay(message: viewModel.processingMessage)
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

class ImportViewModel: ObservableObject {
    @Published var showDocumentPicker = false
    @Published var showCamera = false
    @Published var showPhotoPicker = false
    @Published var selectedPhoto: PhotosPickerItem?
    @Published var isProcessing = false
    @Published var processingMessage = ""

    // TODO: Implement import and processing logic
}

#Preview {
    ImportView()
}
