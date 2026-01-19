import SwiftUI

struct HomeView: View {
    @StateObject private var viewModel = HomeViewModel()
    @State private var showRecordSheet = false
    @State private var showImportSheet = false
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Heart Rate Card
                    HeartRateSummaryCard(
                        averageHR: viewModel.averageHeartRate,
                        minHR: viewModel.minHeartRate,
                        maxHR: viewModel.maxHeartRate
                    )

                    // Quick Actions
                    HStack(spacing: 16) {
                        QuickActionButton(
                            title: String(localized: "home.quickActions.record"),
                            icon: "waveform.path.ecg",
                            color: .red
                        ) {
                            showRecordSheet = true
                        }

                        QuickActionButton(
                            title: String(localized: "home.quickActions.import"),
                            icon: "doc.fill",
                            color: .blue
                        ) {
                            showImportSheet = true
                        }
                    }
                    .padding(.horizontal)

                    // Recent Screenings
                    VStack(alignment: .leading, spacing: 12) {
                        Text("home.recentScreenings")
                            .font(.headline)
                            .padding(.horizontal)

                        if viewModel.recentScreenings.isEmpty {
                            ContentUnavailableView(
                                String(localized: "history.empty"),
                                systemImage: "waveform.path.ecg",
                                description: Text("history.empty.desc")
                            )
                            .frame(height: 200)
                        } else {
                            ForEach(viewModel.recentScreenings) { screening in
                                NavigationLink(destination: ResultDetailView(screening: screening)) {
                                    ScreeningRow(screening: screening)
                                }
                            }
                            .padding(.horizontal)

                            NavigationLink(destination: HistoryView()) {
                                Text("home.seeAll")
                                    .font(.subheadline)
                                    .foregroundStyle(.blue)
                            }
                            .padding(.horizontal)
                        }
                    }

                    Spacer(minLength: 50)
                }
                .padding(.top)
            }
            .navigationTitle("home.title")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showRecordSheet) {
                RecordView()
            }
            .sheet(isPresented: $showImportSheet) {
                ImportView()
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
            .onAppear {
                viewModel.loadData()
            }
        }
    }
}

// MARK: - Heart Rate Summary Card
struct HeartRateSummaryCard: View {
    let averageHR: Int?
    let minHR: Int?
    let maxHR: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("home.heartRate.title")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack(alignment: .firstTextBaseline) {
                Image(systemName: "heart.fill")
                    .foregroundStyle(.red)
                    .font(.title2)

                if let avg = averageHR {
                    Text("\(avg)")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                    Text("bpm")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("(\(String(localized: "home.heartRate.avg")))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("--")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }

            if let min = minHR, let max = maxHR {
                Text(String(format: String(localized: "home.heartRate.range"), min, max))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
        .padding(.horizontal)
    }
}

// MARK: - Quick Action Button
struct QuickActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 32))
                    .foregroundStyle(color)

                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 24)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Screening Row
struct ScreeningRow: View {
    let screening: ScreeningResultModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(screening.date, style: .date)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(screening.primaryCondition)
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
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Preview
#Preview {
    HomeView()
}
