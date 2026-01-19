import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView()
                .tabItem {
                    Label("home.tab", systemImage: "house.fill")
                }
                .tag(0)

            HistoryView()
                .tabItem {
                    Label("history.tab", systemImage: "clock.fill")
                }
                .tag(1)

            ProfileView()
                .tabItem {
                    Label("profile.tab", systemImage: "person.fill")
                }
                .tag(2)
        }
        .tint(.red)
    }
}

#Preview {
    ContentView()
}
