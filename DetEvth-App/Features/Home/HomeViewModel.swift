import Foundation
import Combine

// Temporary model for preview
struct ScreeningResultModel: Identifiable {
    let id: UUID
    let date: Date
    let primaryCondition: String
    let confidence: Float
    let source: String
}

@MainActor
class HomeViewModel: ObservableObject {
    @Published var averageHeartRate: Int?
    @Published var minHeartRate: Int?
    @Published var maxHeartRate: Int?
    @Published var recentScreenings: [ScreeningResultModel] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private var cancellables = Set<AnyCancellable>()

    func loadData() {
        loadHeartRateData()
        loadRecentScreenings()
    }

    private func loadHeartRateData() {
        // TODO: Implement HealthKit heart rate query
        // For now, use placeholder data
        averageHeartRate = 72
        minHeartRate = 58
        maxHeartRate = 92
    }

    private func loadRecentScreenings() {
        // TODO: Implement Core Data fetch
        // For now, use placeholder data
        recentScreenings = [
            ScreeningResultModel(
                id: UUID(),
                date: Date(),
                primaryCondition: "Normal Sinus Rhythm",
                confidence: 0.92,
                source: "healthkit"
            ),
            ScreeningResultModel(
                id: UUID(),
                date: Date().addingTimeInterval(-86400),
                primaryCondition: "Sinus Rhythm",
                confidence: 0.89,
                source: "import"
            ),
            ScreeningResultModel(
                id: UUID(),
                date: Date().addingTimeInterval(-172800),
                primaryCondition: "Sinus Tachycardia",
                confidence: 0.85,
                source: "healthkit"
            )
        ]
    }
}
