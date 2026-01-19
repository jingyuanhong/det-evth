import SwiftUI

struct HistoryView: View {
    @StateObject private var viewModel = HistoryViewModel()
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.screenings.isEmpty {
                    ContentUnavailableView(
                        String(localized: "history.empty"),
                        systemImage: "waveform.path.ecg",
                        description: Text("history.empty.desc")
                    )
                } else {
                    List {
                        ForEach(viewModel.filteredScreenings(searchText: searchText)) { screening in
                            NavigationLink(destination: ResultDetailView(screening: screening)) {
                                HistoryRowView(screening: screening)
                            }
                        }
                        .onDelete(perform: viewModel.deleteScreenings)
                    }
                    .searchable(text: $searchText)
                }
            }
            .navigationTitle("history.title")
            .onAppear {
                viewModel.loadScreenings()
            }
        }
    }
}

struct HistoryRowView: View {
    let screening: ScreeningResultModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(screening.date, style: .date)
                    .font(.headline)

                Spacer()

                Label(
                    screening.source == "healthkit" ?
                        String(localized: "history.source.healthkit") :
                        String(localized: "history.source.import"),
                    systemImage: screening.source == "healthkit" ? "applewatch" : "doc.fill"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Text(screening.primaryCondition)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                Text(String(localized: "results.confidence", defaultValue: "Confidence: \(String(format: "%.1f", screening.confidence * 100))%"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

class HistoryViewModel: ObservableObject {
    @Published var screenings: [ScreeningResultModel] = []

    func loadScreenings() {
        // TODO: Implement Core Data fetch
        screenings = [
            ScreeningResultModel(id: UUID(), date: Date(), primaryCondition: "Normal Sinus Rhythm", confidence: 0.92, source: "healthkit"),
            ScreeningResultModel(id: UUID(), date: Date().addingTimeInterval(-86400), primaryCondition: "Sinus Rhythm", confidence: 0.89, source: "import"),
            ScreeningResultModel(id: UUID(), date: Date().addingTimeInterval(-172800), primaryCondition: "Sinus Tachycardia", confidence: 0.85, source: "healthkit"),
        ]
    }

    func filteredScreenings(searchText: String) -> [ScreeningResultModel] {
        if searchText.isEmpty {
            return screenings
        }
        return screenings.filter { $0.primaryCondition.localizedCaseInsensitiveContains(searchText) }
    }

    func deleteScreenings(at offsets: IndexSet) {
        screenings.remove(atOffsets: offsets)
        // TODO: Delete from Core Data
    }
}

#Preview {
    HistoryView()
}
