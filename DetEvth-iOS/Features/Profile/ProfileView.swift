import SwiftUI

struct ProfileView: View {
    @StateObject private var viewModel = ProfileViewModel()

    var body: some View {
        NavigationStack {
            List {
                // Personal Info Section
                Section {
                    ProfileInfoRow(label: String(localized: "profile.dob"), value: viewModel.dateOfBirth)
                    ProfileInfoRow(label: String(localized: "profile.gender"), value: viewModel.gender)
                    ProfileInfoRow(label: String(localized: "profile.height"), value: viewModel.height)
                    ProfileInfoRow(label: String(localized: "profile.weight"), value: viewModel.weight)
                } header: {
                    Text("profile.personalInfo")
                }

                // Emergency Contacts Section
                Section {
                    ForEach(viewModel.emergencyContacts) { contact in
                        EmergencyContactRow(contact: contact)
                    }

                    Button {
                        viewModel.showAddContact = true
                    } label: {
                        Label(String(localized: "profile.addContact"), systemImage: "plus.circle.fill")
                    }
                } header: {
                    Text("profile.emergencyContacts")
                }

                // Doctor Email Section
                Section {
                    HStack {
                        Text(viewModel.doctorEmail.isEmpty ? "Not set" : viewModel.doctorEmail)
                            .foregroundStyle(viewModel.doctorEmail.isEmpty ? .secondary : .primary)
                        Spacer()
                        Button("Edit") {
                            viewModel.showEditDoctorEmail = true
                        }
                        .font(.caption)
                    }
                } header: {
                    Text("profile.doctorEmail")
                }
            }
            .navigationTitle("profile.title")
            .sheet(isPresented: $viewModel.showAddContact) {
                AddContactView()
            }
        }
    }
}

struct ProfileInfoRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
    }
}

struct EmergencyContact: Identifiable {
    let id: UUID
    let name: String
    let phone: String
    let relationship: String
}

struct EmergencyContactRow: View {
    let contact: EmergencyContact

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(contact.name)
                    .font(.headline)
                Text(contact.relationship)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                // Call action
                if let url = URL(string: "tel://\(contact.phone)") {
                    UIApplication.shared.open(url)
                }
            } label: {
                Label(String(localized: "profile.call"), systemImage: "phone.fill")
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
        }
    }
}

struct AddContactView: View {
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            Text("Add Contact Form")
                .navigationTitle("profile.addContact")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("common.cancel") { dismiss() }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("common.save") { dismiss() }
                    }
                }
        }
    }
}

class ProfileViewModel: ObservableObject {
    @Published var dateOfBirth = "1990-01-01"
    @Published var gender = "Male"
    @Published var height = "175 cm"
    @Published var weight = "70 kg"
    @Published var doctorEmail = ""
    @Published var emergencyContacts: [EmergencyContact] = [
        EmergencyContact(id: UUID(), name: "Dr. Smith", phone: "1234567890", relationship: "Cardiologist"),
        EmergencyContact(id: UUID(), name: "Jane Doe", phone: "0987654321", relationship: "Spouse")
    ]
    @Published var showAddContact = false
    @Published var showEditDoctorEmail = false
}

#Preview {
    ProfileView()
}
