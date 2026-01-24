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
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    viewModel.deleteScreening(screening)
                                } label: {
                                    Text("common.delete")
                                }
                            }
                        }
                    }
                    .searchable(text: $searchText)
                }
            }
            .navigationTitle("history.title")
            .onAppear {
                viewModel.loadScreenings()
            }
            .refreshable {
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
                Text(String(format: String(localized: "results.confidence"), screening.confidence * 100))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

class HistoryViewModel: ObservableObject {
    @Published var screenings: [ScreeningResultModel] = []

    private let repository = ScreeningRepository.shared
    private var screeningEntities: [ScreeningResultEntity] = []

    func loadScreenings() {
        screeningEntities = repository.fetchAllScreeningResults()
        screenings = screeningEntities.map { entity in
            // Try to extract heart rate from signal if available
            var heartRate: Int? = nil
            if let signal = entity.ecgRecord?.signal, !signal.isEmpty {
                let sampleRate = entity.ecgRecord?.sampleRate ?? 500
                heartRate = ECGHeartRateExtractor.extractHeartRate(from: signal, sampleRate: sampleRate)
            }

            return ScreeningResultModel(
                id: entity.id ?? UUID(),
                date: entity.createdAt ?? Date(),
                primaryCondition: entity.primaryCondition?.localizedName ?? String(localized: "common.unknown"),
                confidence: entity.primaryConfidence,
                source: entity.ecgRecord?.source ?? "import",
                signal: entity.ecgRecord?.signal,
                probabilities: entity.probabilities.isEmpty ? nil : entity.probabilities,
                heartRate: heartRate
            )
        }
    }

    func filteredScreenings(searchText: String) -> [ScreeningResultModel] {
        if searchText.isEmpty {
            return screenings
        }
        return screenings.filter { $0.primaryCondition.localizedCaseInsensitiveContains(searchText) }
    }

    func deleteScreening(_ screening: ScreeningResultModel) {
        // Find and delete from Core Data
        if let entityIndex = screeningEntities.firstIndex(where: { $0.id == screening.id }) {
            let entity = screeningEntities[entityIndex]
            repository.deleteScreeningResult(entity)
            screeningEntities.remove(at: entityIndex)
        }
        // Reload to refresh the UI
        loadScreenings()
    }
}

#Preview {
    HistoryView()
}
