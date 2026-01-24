import SwiftUI

struct ProfileView: View {
    @StateObject private var viewModel = ProfileViewModel()
    @Environment(\.editMode) private var editMode

    var body: some View {
        NavigationStack {
            List {
                // Personal Info Section
                Section {
                    Button {
                        viewModel.showEditProfile = true
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 8) {
                                ProfileInfoRow(label: String(localized: "profile.dob"), value: viewModel.dateOfBirth)
                                ProfileInfoRow(label: String(localized: "profile.gender"), value: viewModel.gender)
                                ProfileInfoRow(label: String(localized: "profile.height"), value: viewModel.height)
                                ProfileInfoRow(label: String(localized: "profile.weight"), value: viewModel.weight)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                } header: {
                    Text("profile.personalInfo")
                        .textCase(nil)
                }

                // Emergency Contacts Section
                Section {
                    ForEach(viewModel.emergencyContacts) { contact in
                        EmergencyContactRow(
                            contact: contact,
                            onEdit: { viewModel.editContact(contact) },
                            onCall: { viewModel.callContact(contact) }
                        )
                    }
                    .onDelete(perform: viewModel.deleteContacts)

                    Button {
                        viewModel.showAddContact = true
                    } label: {
                        Label(String(localized: "profile.addContact"), systemImage: "plus.circle.fill")
                    }
                } header: {
                    Text("profile.emergencyContacts")
                        .textCase(nil)
                }

                // Doctor Email Section
                Section {
                    Button {
                        viewModel.showEditDoctorEmail = true
                    } label: {
                        HStack {
                            Text(viewModel.doctorEmail.isEmpty ? String(localized: "profile.notSet") : viewModel.doctorEmail)
                                .foregroundStyle(viewModel.doctorEmail.isEmpty ? .secondary : .primary)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                } header: {
                    Text("profile.doctorEmail")
                        .textCase(nil)
                }
            }
            .navigationTitle("profile.title")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if !viewModel.emergencyContacts.isEmpty {
                        Button {
                            withAnimation {
                                editMode?.wrappedValue = editMode?.wrappedValue == .active ? .inactive : .active
                            }
                        } label: {
                            Text(editMode?.wrappedValue == .active ? "common.done" : "common.edit")
                        }
                    }
                }
            }
            .sheet(isPresented: $viewModel.showEditProfile) {
                EditProfileView(viewModel: viewModel)
            }
            .sheet(isPresented: $viewModel.showAddContact) {
                AddContactView(viewModel: viewModel, contactToEdit: nil)
            }
            .sheet(isPresented: $viewModel.showEditContact) {
                if let contact = viewModel.selectedContact {
                    AddContactView(viewModel: viewModel, contactToEdit: contact)
                }
            }
            .sheet(isPresented: $viewModel.showEditDoctorEmail) {
                EditDoctorEmailView(viewModel: viewModel)
            }
            .onAppear {
                viewModel.loadProfile()
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
    var name: String
    var phone: String
    var relationship: String
}

struct EmergencyContactRow: View {
    let contact: EmergencyContact
    let onEdit: () -> Void
    let onCall: () -> Void

    var body: some View {
        HStack {
            Button(action: onEdit) {
                VStack(alignment: .leading) {
                    Text(contact.name)
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Text(contact.relationship)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)

            Spacer()

            Button(action: onCall) {
                Label(String(localized: "profile.call"), systemImage: "phone.fill")
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
        }
    }
}

// MARK: - Edit Profile View

struct EditProfileView: View {
    @ObservedObject var viewModel: ProfileViewModel
    @Environment(\.dismiss) var dismiss

    @State private var dateOfBirth: Date = Date()
    @State private var gender: String = ""
    @State private var heightCM: String = ""
    @State private var weightKG: String = ""

    let genderOptions = ["Male", "Female", "Other"]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    DatePicker(
                        String(localized: "profile.dob"),
                        selection: $dateOfBirth,
                        displayedComponents: .date
                    )

                    Picker(String(localized: "profile.gender"), selection: $gender) {
                        ForEach(genderOptions, id: \.self) { option in
                            Text(option).tag(option)
                        }
                    }
                }

                Section {
                    HStack {
                        Text(String(localized: "profile.height"))
                        Spacer()
                        TextField("175", text: $heightCM)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Text("cm")
                            .foregroundStyle(.secondary)
                    }

                    HStack {
                        Text(String(localized: "profile.weight"))
                        Spacer()
                        TextField("70", text: $weightKG)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                        Text("kg")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("profile.editProfile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("common.cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("common.save") {
                        viewModel.saveProfile(
                            dateOfBirth: dateOfBirth,
                            gender: gender,
                            heightCM: Float(heightCM) ?? 0,
                            weightKG: Float(weightKG) ?? 0
                        )
                        dismiss()
                    }
                }
            }
            .onAppear {
                // Load current values
                dateOfBirth = viewModel.dateOfBirthDate ?? Date()
                gender = viewModel.gender
                heightCM = viewModel.heightValue > 0 ? String(format: "%.0f", viewModel.heightValue) : ""
                weightKG = viewModel.weightValue > 0 ? String(format: "%.0f", viewModel.weightValue) : ""
            }
        }
    }
}

// MARK: - Add/Edit Contact View

struct AddContactView: View {
    @ObservedObject var viewModel: ProfileViewModel
    let contactToEdit: EmergencyContact?
    @Environment(\.dismiss) var dismiss

    @State private var name: String = ""
    @State private var phone: String = ""
    @State private var relationship: String = ""

    var isEditing: Bool { contactToEdit != nil }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField(String(localized: "profile.contact.name"), text: $name)
                    TextField(String(localized: "profile.contact.phone"), text: $phone)
                        .keyboardType(.phonePad)
                    TextField(String(localized: "profile.contact.relationship"), text: $relationship)
                }
            }
            .navigationTitle(isEditing ? "profile.editContact" : "profile.addContact")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("common.cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("common.save") {
                        if let contact = contactToEdit {
                            viewModel.updateContact(contact, name: name, phone: phone, relationship: relationship)
                        } else {
                            viewModel.addContact(name: name, phone: phone, relationship: relationship)
                        }
                        dismiss()
                    }
                    .disabled(name.isEmpty || phone.isEmpty)
                }
            }
            .onAppear {
                if let contact = contactToEdit {
                    name = contact.name
                    phone = contact.phone
                    relationship = contact.relationship
                }
            }
        }
    }
}

// MARK: - Edit Doctor Email View

struct EditDoctorEmailView: View {
    @ObservedObject var viewModel: ProfileViewModel
    @Environment(\.dismiss) var dismiss

    @State private var email: String = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField(String(localized: "profile.doctorEmail.placeholder"), text: $email)
                        .keyboardType(.emailAddress)
                        .textContentType(.emailAddress)
                        .autocapitalization(.none)
                } footer: {
                    Text("profile.doctorEmail.hint")
                }
            }
            .navigationTitle("profile.doctorEmail")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("common.cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("common.save") {
                        viewModel.saveDoctorEmail(email)
                        dismiss()
                    }
                }
            }
            .onAppear {
                email = viewModel.doctorEmail
            }
        }
    }
}

// MARK: - Profile View Model

class ProfileViewModel: ObservableObject {
    @Published var dateOfBirth = ""
    @Published var dateOfBirthDate: Date?
    @Published var gender = ""
    @Published var height = ""
    @Published var weight = ""
    @Published var heightValue: Float = 0
    @Published var weightValue: Float = 0
    @Published var doctorEmail = ""
    @Published var emergencyContacts: [EmergencyContact] = []

    @Published var showEditProfile = false
    @Published var showAddContact = false
    @Published var showEditContact = false
    @Published var showEditDoctorEmail = false
    @Published var selectedContact: EmergencyContact?

    private let profileRepository = UserProfileRepository.shared
    private let contactsRepository = EmergencyContactsRepository.shared
    private var contactEntities: [EmergencyContactEntity] = []

    func loadProfile() {
        let profile = profileRepository.getProfile()

        // Date of birth
        if let dob = profile.dateOfBirth {
            dateOfBirthDate = dob
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            dateOfBirth = formatter.string(from: dob)
        } else {
            dateOfBirth = String(localized: "profile.notSet")
        }

        // Gender
        gender = profile.gender ?? String(localized: "profile.notSet")

        // Height
        heightValue = profile.heightCM
        if profile.heightCM > 0 {
            height = String(format: "%.0f cm", profile.heightCM)
        } else {
            height = String(localized: "profile.notSet")
        }

        // Weight
        weightValue = profile.weightKG
        if profile.weightKG > 0 {
            weight = String(format: "%.0f kg", profile.weightKG)
        } else {
            weight = String(localized: "profile.notSet")
        }

        // Doctor email
        doctorEmail = profile.doctorEmail ?? ""

        // Load emergency contacts
        loadContacts()
    }

    func loadContacts() {
        contactEntities = contactsRepository.fetchAll()
        emergencyContacts = contactEntities.map { entity in
            EmergencyContact(
                id: entity.id,
                name: entity.name,
                phone: entity.phoneNumber,
                relationship: entity.relationship
            )
        }
    }

    func saveProfile(dateOfBirth: Date, gender: String, heightCM: Float, weightKG: Float) {
        profileRepository.updateProfile(
            dateOfBirth: dateOfBirth,
            gender: gender,
            heightCM: heightCM,
            weightKG: weightKG
        )
        loadProfile()
    }

    func saveDoctorEmail(_ email: String) {
        profileRepository.updateProfile(doctorEmail: email)
        doctorEmail = email
    }

    func addContact(name: String, phone: String, relationship: String) {
        _ = contactsRepository.add(name: name, phone: phone, relationship: relationship)
        loadContacts()
    }

    func editContact(_ contact: EmergencyContact) {
        selectedContact = contact
        showEditContact = true
    }

    func updateContact(_ contact: EmergencyContact, name: String, phone: String, relationship: String) {
        if let entity = contactEntities.first(where: { $0.id == contact.id }) {
            contactsRepository.update(entity, name: name, phone: phone, relationship: relationship)
            loadContacts()
        }
    }

    func deleteContacts(at offsets: IndexSet) {
        for index in offsets {
            if index < emergencyContacts.count {
                let contact = emergencyContacts[index]
                if let entity = contactEntities.first(where: { $0.id == contact.id }) {
                    contactsRepository.delete(entity)
                }
            }
        }
        loadContacts()
    }

    func callContact(_ contact: EmergencyContact) {
        if let url = URL(string: "tel://\(contact.phone)") {
            UIApplication.shared.open(url)
        }
    }
}

#Preview {
    ProfileView()
}
