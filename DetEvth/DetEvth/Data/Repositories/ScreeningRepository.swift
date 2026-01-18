// ScreeningRepository.swift
// Data Repository for ECG Records and Screening Results
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import CoreData
import Combine

/// Repository for managing ECG records and screening results
final class ScreeningRepository: ObservableObject {

    // MARK: - Singleton

    static let shared = ScreeningRepository()

    // MARK: - Properties

    private let persistenceController: PersistenceController
    private var cancellables = Set<AnyCancellable>()

    @Published var recentScreenings: [ScreeningResultEntity] = []

    // MARK: - Initialization

    init(persistenceController: PersistenceController = .shared) {
        self.persistenceController = persistenceController
        loadRecentScreenings()
    }

    // MARK: - ECG Records

    /// Save a new ECG record with signal data
    func saveECGRecord(
        source: ECGSource,
        signal: [Float],
        sampleRate: Float,
        duration: Float,
        fileName: String? = nil
    ) -> ECGRecordEntity {
        let context = persistenceController.viewContext

        let record = ECGRecordEntity(context: context)
        record.id = UUID()
        record.createdAt = Date()
        record.source = source.rawValue
        record.sampleRate = sampleRate
        record.durationSeconds = duration
        record.rawSignalData = signal.withUnsafeBufferPointer { Data(buffer: $0) }
        record.originalFileName = fileName

        persistenceController.save()

        return record
    }

    /// Fetch all ECG records sorted by date
    func fetchAllECGRecords() -> [ECGRecordEntity] {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<ECGRecordEntity>(entityName: "ECGRecordEntity")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \ECGRecordEntity.createdAt, ascending: false)]

        do {
            return try context.fetch(request)
        } catch {
            print("[ScreeningRepository] Fetch error: \(error)")
            return []
        }
    }

    /// Fetch ECG records with optional filtering
    func fetchECGRecords(
        source: ECGSource? = nil,
        startDate: Date? = nil,
        endDate: Date? = nil,
        limit: Int? = nil
    ) -> [ECGRecordEntity] {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<ECGRecordEntity>(entityName: "ECGRecordEntity")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \ECGRecordEntity.createdAt, ascending: false)]

        var predicates: [NSPredicate] = []

        if let source = source {
            predicates.append(NSPredicate(format: "source == %@", source.rawValue))
        }

        if let startDate = startDate {
            predicates.append(NSPredicate(format: "createdAt >= %@", startDate as NSDate))
        }

        if let endDate = endDate {
            predicates.append(NSPredicate(format: "createdAt <= %@", endDate as NSDate))
        }

        if !predicates.isEmpty {
            request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        }

        if let limit = limit {
            request.fetchLimit = limit
        }

        do {
            return try context.fetch(request)
        } catch {
            print("[ScreeningRepository] Fetch error: \(error)")
            return []
        }
    }

    // MARK: - Screening Results

    /// Save screening result for an ECG record
    func saveScreeningResult(
        for ecgRecord: ECGRecordEntity,
        probabilities: [Float],
        perStripProbabilities: [[Float]]? = nil
    ) -> ScreeningResultEntity {
        let context = persistenceController.viewContext

        let result = ScreeningResultEntity(context: context)
        result.id = UUID()
        result.createdAt = Date()
        result.modelVersion = Constants.App.modelVersion
        result.probabilitiesData = probabilities.withUnsafeBufferPointer { Data(buffer: $0) }

        // Store per-strip probabilities if provided
        if let perStrip = perStripProbabilities {
            let flattened = perStrip.flatMap { $0 }
            result.perStripProbabilitiesData = flattened.withUnsafeBufferPointer { Data(buffer: $0) }
        }

        // Find primary condition
        if let maxIndex = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) {
            result.primaryConditionIndex = Int16(maxIndex)
            result.primaryConfidence = probabilities[maxIndex]
        }

        result.ecgRecord = ecgRecord
        ecgRecord.screeningResult = result

        persistenceController.save()
        loadRecentScreenings()

        return result
    }

    /// Fetch all screening results sorted by date
    func fetchAllScreeningResults() -> [ScreeningResultEntity] {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<ScreeningResultEntity>(entityName: "ScreeningResultEntity")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \ScreeningResultEntity.createdAt, ascending: false)]

        do {
            return try context.fetch(request)
        } catch {
            print("[ScreeningRepository] Fetch error: \(error)")
            return []
        }
    }

    /// Fetch recent screening results
    func fetchRecentScreeningResults(limit: Int = 10) -> [ScreeningResultEntity] {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<ScreeningResultEntity>(entityName: "ScreeningResultEntity")
        request.sortDescriptors = [NSSortDescriptor(keyPath: \ScreeningResultEntity.createdAt, ascending: false)]
        request.fetchLimit = limit

        do {
            return try context.fetch(request)
        } catch {
            print("[ScreeningRepository] Fetch error: \(error)")
            return []
        }
    }

    /// Search screening results by condition
    func searchScreeningResults(query: String) -> [ScreeningResultEntity] {
        // First fetch all results, then filter by condition name
        // (In production, you might want to add a denormalized searchable field)
        let allResults = fetchAllScreeningResults()

        guard !query.isEmpty else { return allResults }

        return allResults.filter { result in
            guard let condition = result.primaryCondition else { return false }
            return condition.nameEN.localizedCaseInsensitiveContains(query) ||
                   condition.nameCN.contains(query)
        }
    }

    // MARK: - Delete

    /// Delete an ECG record and its associated screening result
    func deleteECGRecord(_ record: ECGRecordEntity) {
        let context = persistenceController.viewContext
        context.delete(record)
        persistenceController.save()
        loadRecentScreenings()
    }

    /// Delete a screening result
    func deleteScreeningResult(_ result: ScreeningResultEntity) {
        let context = persistenceController.viewContext
        context.delete(result)
        persistenceController.save()
        loadRecentScreenings()
    }

    // MARK: - Private

    private func loadRecentScreenings() {
        recentScreenings = fetchRecentScreeningResults(limit: 10)
    }
}

// MARK: - Emergency Contacts Repository

final class EmergencyContactsRepository {

    static let shared = EmergencyContactsRepository()

    private let persistenceController: PersistenceController

    init(persistenceController: PersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    /// Fetch all emergency contacts
    func fetchAll() -> [EmergencyContactEntity] {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<EmergencyContactEntity>(entityName: "EmergencyContactEntity")
        request.sortDescriptors = [
            NSSortDescriptor(keyPath: \EmergencyContactEntity.sortOrder, ascending: true),
            NSSortDescriptor(keyPath: \EmergencyContactEntity.name, ascending: true)
        ]

        do {
            return try context.fetch(request)
        } catch {
            print("[EmergencyContactsRepository] Fetch error: \(error)")
            return []
        }
    }

    /// Add a new emergency contact
    func add(name: String, phone: String, relationship: String, email: String? = nil, isDoctor: Bool = false) -> EmergencyContactEntity {
        let context = persistenceController.viewContext

        let contact = EmergencyContactEntity(context: context)
        contact.id = UUID()
        contact.name = name
        contact.phoneNumber = phone
        contact.relationship = relationship
        contact.email = email
        contact.isDoctor = isDoctor
        contact.sortOrder = Int16(fetchAll().count)
        contact.createdAt = Date()

        persistenceController.save()

        return contact
    }

    /// Update an emergency contact
    func update(_ contact: EmergencyContactEntity, name: String? = nil, phone: String? = nil, relationship: String? = nil, email: String? = nil) {
        if let name = name { contact.name = name }
        if let phone = phone { contact.phoneNumber = phone }
        if let relationship = relationship { contact.relationship = relationship }
        if let email = email { contact.email = email }

        persistenceController.save()
    }

    /// Delete an emergency contact
    func delete(_ contact: EmergencyContactEntity) {
        let context = persistenceController.viewContext
        context.delete(contact)
        persistenceController.save()
    }

    /// Get doctor contacts only
    func fetchDoctors() -> [EmergencyContactEntity] {
        return fetchAll().filter { $0.isDoctor }
    }
}

// MARK: - User Profile Repository

final class UserProfileRepository {

    static let shared = UserProfileRepository()

    private let persistenceController: PersistenceController

    init(persistenceController: PersistenceController = .shared) {
        self.persistenceController = persistenceController
    }

    /// Get or create the user profile
    func getProfile() -> UserProfileEntity {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<UserProfileEntity>(entityName: "UserProfileEntity")
        request.fetchLimit = 1

        do {
            if let existing = try context.fetch(request).first {
                return existing
            }
        } catch {
            print("[UserProfileRepository] Fetch error: \(error)")
        }

        // Create new profile
        let profile = UserProfileEntity(context: context)
        profile.id = UUID()
        profile.heightCM = 0
        profile.weightKG = 0
        profile.createdAt = Date()
        profile.updatedAt = Date()

        persistenceController.save()

        return profile
    }

    /// Update user profile
    func updateProfile(
        dateOfBirth: Date? = nil,
        gender: String? = nil,
        heightCM: Float? = nil,
        weightKG: Float? = nil,
        doctorEmail: String? = nil
    ) {
        let profile = getProfile()

        if let dob = dateOfBirth { profile.dateOfBirth = dob }
        if let gender = gender { profile.gender = gender }
        if let height = heightCM { profile.heightCM = height }
        if let weight = weightKG { profile.weightKG = weight }
        if let email = doctorEmail { profile.doctorEmail = email }

        profile.updatedAt = Date()

        persistenceController.save()
    }
}
