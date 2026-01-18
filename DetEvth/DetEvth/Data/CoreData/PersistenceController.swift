// PersistenceController.swift
// Core Data Stack for DetEvth
// © 2026 minuscule health Ltd. All rights reserved.

import CoreData

struct PersistenceController {
    static let shared = PersistenceController()

    let container: NSPersistentContainer

    var viewContext: NSManagedObjectContext {
        container.viewContext
    }

    init(inMemory: Bool = false) {
        // Create container with programmatic model
        let model = NSManagedObjectModel.createDetEvthModel()
        container = NSPersistentContainer(name: "DetEvth", managedObjectModel: model)

        if inMemory {
            container.persistentStoreDescriptions.first!.url = URL(fileURLWithPath: "/dev/null")
        }

        container.loadPersistentStores { storeDescription, error in
            if let error = error as NSError? {
                // In production, handle this gracefully
                print("[PersistenceController] Core Data error: \(error), \(error.userInfo)")
                #if DEBUG
                fatalError("Core Data error: \(error), \(error.userInfo)")
                #endif
            }
            print("[PersistenceController] Loaded store: \(storeDescription.url?.lastPathComponent ?? "unknown")")
        }

        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
    }

    // MARK: - Background Context

    func newBackgroundContext() -> NSManagedObjectContext {
        let context = container.newBackgroundContext()
        context.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
        return context
    }

    // MARK: - Save

    func save() {
        let context = container.viewContext
        guard context.hasChanges else { return }

        do {
            try context.save()
        } catch {
            print("[PersistenceController] Save error: \(error)")
        }
    }

    func saveContext(_ context: NSManagedObjectContext) {
        guard context.hasChanges else { return }

        do {
            try context.save()
        } catch {
            print("[PersistenceController] Save error: \(error)")
        }
    }

    // MARK: - Preview Helper

    static var preview: PersistenceController = {
        let result = PersistenceController(inMemory: true)
        let viewContext = result.container.viewContext

        // Create sample ECG records with screening results
        let sampleConditions = [
            ("Normal Sinus Rhythm", Float(0.92)),
            ("Sinus Bradycardia", Float(0.78)),
            ("Atrial Fibrillation", Float(0.85)),
            ("Left Bundle Branch Block", Float(0.67)),
            ("Sinus Tachycardia", Float(0.73))
        ]

        for (i, (condition, confidence)) in sampleConditions.enumerated() {
            // Create ECG record
            let ecgRecord = ECGRecordEntity(context: viewContext)
            ecgRecord.id = UUID()
            ecgRecord.createdAt = Date().addingTimeInterval(Double(-i * 86400))
            ecgRecord.source = i % 2 == 0 ? ECGSource.healthkit.rawValue : ECGSource.importPDF.rawValue
            ecgRecord.sampleRate = 500
            ecgRecord.durationSeconds = 30

            // Create screening result
            let screeningResult = ScreeningResultEntity(context: viewContext)
            screeningResult.id = UUID()
            screeningResult.createdAt = ecgRecord.createdAt
            screeningResult.modelVersion = Constants.App.modelVersion

            // Create sample probabilities
            var probabilities = [Float](repeating: 0.05, count: 150)
            probabilities[0] = confidence  // Set primary condition

            screeningResult.probabilitiesData = probabilities.withUnsafeBufferPointer { Data(buffer: $0) }
            screeningResult.primaryConditionIndex = 0
            screeningResult.primaryConfidence = confidence

            ecgRecord.screeningResult = screeningResult
        }

        // Create sample emergency contacts
        let contacts = [
            ("Dr. Smith", "1234567890", "Cardiologist", true),
            ("Jane Doe", "0987654321", "Spouse", false)
        ]

        for (name, phone, relationship, isDoctor) in contacts {
            let contact = EmergencyContactEntity(context: viewContext)
            contact.id = UUID()
            contact.name = name
            contact.phoneNumber = phone
            contact.relationship = relationship
            contact.isDoctor = isDoctor
            contact.sortOrder = 0
            contact.createdAt = Date()
        }

        // Create sample user profile
        let profile = UserProfileEntity(context: viewContext)
        profile.id = UUID()
        profile.dateOfBirth = Calendar.current.date(byAdding: .year, value: -35, to: Date())
        profile.gender = "Male"
        profile.heightCM = 175
        profile.weightKG = 70
        profile.createdAt = Date()
        profile.updatedAt = Date()

        do {
            try viewContext.save()
        } catch {
            let nsError = error as NSError
            print("Preview data error: \(nsError), \(nsError.userInfo)")
        }

        return result
    }()
}
