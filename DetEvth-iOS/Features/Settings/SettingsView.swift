import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) var dismiss
    @AppStorage("selectedLanguage") private var selectedLanguage = "en"
    @AppStorage("notificationsEnabled") private var notificationsEnabled = true

    var body: some View {
        NavigationStack {
            List {
                // Language Section
                Section {
                    Picker(String(localized: "settings.language"), selection: $selectedLanguage) {
                        Text("settings.language.en").tag("en")
                        Text("settings.language.cn").tag("zh-Hans")
                    }
                }

                // Notifications Section
                Section {
                    Toggle(String(localized: "settings.notifications"), isOn: $notificationsEnabled)
                }

                // Legal Section
                Section {
                    NavigationLink {
                        LegalDocumentView(title: "settings.privacy", content: LegalDocuments.privacyPolicy)
                    } label: {
                        Text("settings.privacy")
                    }

                    NavigationLink {
                        LegalDocumentView(title: "settings.terms", content: LegalDocuments.termsOfService)
                    } label: {
                        Text("settings.terms")
                    }
                }

                // About Section
                Section {
                    HStack {
                        Text("settings.version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundStyle(.secondary)
                    }

                    VStack(spacing: 4) {
                        Image("CompanyLogo")
                            .resizable()
                            .scaledToFit()
                            .frame(height: 40)

                        Text("minuscule health ltd.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Text("© 2026")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                } header: {
                    Text("settings.about")
                }
            }
            .navigationTitle("settings.title")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("common.done") { dismiss() }
                }
            }
        }
    }
}

struct LegalDocumentView: View {
    let title: LocalizedStringKey
    let content: String

    var body: some View {
        ScrollView {
            Text(content)
                .font(.body)
                .padding()
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    SettingsView()
}
