// CoreDataModels.swift
// Core Data Entity Definitions for DetEvth
// © 2026 minuscule health Ltd. All rights reserved.

import Foundation
import CoreData

// MARK: - ECGRecord Entity

/// Core Data entity for storing ECG recordings
@objc(ECGRecordEntity)
public class ECGRecordEntity: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var createdAt: Date
    @NSManaged public var source: String  // "healthkit", "import_pdf", "import_image"
    @NSManaged public var sampleRate: Float
    @NSManaged public var durationSeconds: Float
    @NSManaged public var rawSignalData: Data?  // Compressed signal data
    @NSManaged public var originalFileName: String?  // For imported files
    @NSManaged public var notes: String?
    @NSManaged public var screeningResult: ScreeningResultEntity?

    // Convenience initializer
    static func create(
        in context: NSManagedObjectContext,
        source: ECGSource,
        sampleRate: Float,
        duration: Float,
        signal: [Float]
    ) -> ECGRecordEntity {
        let entity = ECGRecordEntity(context: context)
        entity.id = UUID()
        entity.createdAt = Date()
        entity.source = source.rawValue
        entity.sampleRate = sampleRate
        entity.durationSeconds = duration
        entity.rawSignalData = signal.withUnsafeBufferPointer { Data(buffer: $0) }
        return entity
    }

    // Get signal as Float array
    var signal: [Float]? {
        guard let data = rawSignalData else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }
}

// MARK: - ScreeningResult Entity

/// Core Data entity for storing screening results
@objc(ScreeningResultEntity)
public class ScreeningResultEntity: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var createdAt: Date
    @NSManaged public var primaryConditionIndex: Int16
    @NSManaged public var primaryConfidence: Float
    @NSManaged public var probabilitiesData: Data?  // 150 floats serialized
    @NSManaged public var perStripProbabilitiesData: Data?  // For multi-strip averaging
    @NSManaged public var modelVersion: String
    @NSManaged public var ecgRecord: ECGRecordEntity?

    // Convenience initializer
    static func create(
        in context: NSManagedObjectContext,
        probabilities: [Float],
        ecgRecord: ECGRecordEntity?
    ) -> ScreeningResultEntity {
        let entity = ScreeningResultEntity(context: context)
        entity.id = UUID()
        entity.createdAt = Date()
        entity.modelVersion = Constants.App.modelVersion

        // Store probabilities
        entity.probabilitiesData = probabilities.withUnsafeBufferPointer { Data(buffer: $0) }

        // Find primary condition (excluding index 0 "abnormal ecg" - too generic)
        let validIndices = probabilities.indices.filter { $0 != 0 }
        if let maxIndex = validIndices.max(by: { probabilities[$0] < probabilities[$1] }) {
            entity.primaryConditionIndex = Int16(maxIndex)
            entity.primaryConfidence = probabilities[maxIndex]
        }

        entity.ecgRecord = ecgRecord
        return entity
    }

    // Get probabilities as Float array
    var probabilities: [Float] {
        guard let data = probabilitiesData else { return [] }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }

    // Get primary condition
    var primaryCondition: DiseaseCondition? {
        let index = Int(primaryConditionIndex)
        guard index >= 0 && index < DiseaseConditions.all.count else { return nil }
        return DiseaseConditions.all[index]
    }

    // Get top conditions (excluding index 0 "abnormal ecg" - too generic)
    func topConditions(count: Int = 10, threshold: Float = 0.1) -> [(condition: DiseaseCondition, probability: Float)] {
        let probs = probabilities
        let conditions = DiseaseConditions.all

        var results: [(DiseaseCondition, Float)] = []
        for (index, prob) in probs.enumerated() where index != 0 && prob >= threshold && index < conditions.count {
            results.append((conditions[index], prob))
        }

        results.sort { $0.1 > $1.1 }
        return Array(results.prefix(count))
    }
}

// MARK: - EmergencyContact Entity

/// Core Data entity for storing emergency contacts
@objc(EmergencyContactEntity)
public class EmergencyContactEntity: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var name: String
    @NSManaged public var phoneNumber: String
    @NSManaged public var relationship: String
    @NSManaged public var email: String?
    @NSManaged public var isDoctor: Bool
    @NSManaged public var sortOrder: Int16
    @NSManaged public var createdAt: Date

    // Convenience initializer
    static func create(
        in context: NSManagedObjectContext,
        name: String,
        phone: String,
        relationship: String,
        isDoctor: Bool = false
    ) -> EmergencyContactEntity {
        let entity = EmergencyContactEntity(context: context)
        entity.id = UUID()
        entity.name = name
        entity.phoneNumber = phone
        entity.relationship = relationship
        entity.isDoctor = isDoctor
        entity.sortOrder = 0
        entity.createdAt = Date()
        return entity
    }
}

// MARK: - UserProfile Entity

/// Core Data entity for storing user profile
@objc(UserProfileEntity)
public class UserProfileEntity: NSManagedObject {

    @NSManaged public var id: UUID
    @NSManaged public var dateOfBirth: Date?
    @NSManaged public var gender: String?
    @NSManaged public var heightCM: Float
    @NSManaged public var weightKG: Float
    @NSManaged public var doctorEmail: String?
    @NSManaged public var createdAt: Date
    @NSManaged public var updatedAt: Date

    // Convenience initializer
    static func create(in context: NSManagedObjectContext) -> UserProfileEntity {
        let entity = UserProfileEntity(context: context)
        entity.id = UUID()
        entity.createdAt = Date()
        entity.updatedAt = Date()
        return entity
    }
}

// MARK: - ECG Source Enum

enum ECGSource: String, CaseIterable {
    case healthkit = "healthkit"
    case importPDF = "import_pdf"
    case importImage = "import_image"
    case camera = "camera"
    case appleWatch = "apple_watch"

    var displayName: String {
        switch self {
        case .healthkit: return String(localized: "source.healthkit")
        case .importPDF: return String(localized: "source.pdf")
        case .importImage: return String(localized: "source.image")
        case .camera: return String(localized: "source.camera")
        case .appleWatch: return String(localized: "source.appleWatch")
        }
    }

    var iconName: String {
        switch self {
        case .healthkit: return "heart.fill"
        case .importPDF: return "doc.fill"
        case .importImage: return "photo.fill"
        case .camera: return "camera.fill"
        case .appleWatch: return "applewatch"
        }
    }
}

// MARK: - Core Data Model Creation

extension NSManagedObjectModel {

    /// Create the Core Data model programmatically
    static func createDetEvthModel() -> NSManagedObjectModel {
        let model = NSManagedObjectModel()

        // ECGRecord Entity
        let ecgRecordEntity = NSEntityDescription()
        ecgRecordEntity.name = "ECGRecordEntity"
        ecgRecordEntity.managedObjectClassName = "ECGRecordEntity"

        let ecgRecordAttributes: [(String, NSAttributeType, Bool)] = [
            ("id", .UUIDAttributeType, false),
            ("createdAt", .dateAttributeType, false),
            ("source", .stringAttributeType, false),
            ("sampleRate", .floatAttributeType, false),
            ("durationSeconds", .floatAttributeType, false),
            ("rawSignalData", .binaryDataAttributeType, true),
            ("originalFileName", .stringAttributeType, true),
            ("notes", .stringAttributeType, true)
        ]

        ecgRecordEntity.properties = ecgRecordAttributes.map { name, type, optional in
            let attr = NSAttributeDescription()
            attr.name = name
            attr.attributeType = type
            attr.isOptional = optional
            return attr
        }

        // ScreeningResult Entity
        let screeningResultEntity = NSEntityDescription()
        screeningResultEntity.name = "ScreeningResultEntity"
        screeningResultEntity.managedObjectClassName = "ScreeningResultEntity"

        let screeningResultAttributes: [(String, NSAttributeType, Bool)] = [
            ("id", .UUIDAttributeType, false),
            ("createdAt", .dateAttributeType, false),
            ("primaryConditionIndex", .integer16AttributeType, false),
            ("primaryConfidence", .floatAttributeType, false),
            ("probabilitiesData", .binaryDataAttributeType, true),
            ("perStripProbabilitiesData", .binaryDataAttributeType, true),
            ("modelVersion", .stringAttributeType, false)
        ]

        screeningResultEntity.properties = screeningResultAttributes.map { name, type, optional in
            let attr = NSAttributeDescription()
            attr.name = name
            attr.attributeType = type
            attr.isOptional = optional
            return attr
        }

        // EmergencyContact Entity
        let emergencyContactEntity = NSEntityDescription()
        emergencyContactEntity.name = "EmergencyContactEntity"
        emergencyContactEntity.managedObjectClassName = "EmergencyContactEntity"

        let emergencyContactAttributes: [(String, NSAttributeType, Bool)] = [
            ("id", .UUIDAttributeType, false),
            ("name", .stringAttributeType, false),
            ("phoneNumber", .stringAttributeType, false),
            ("relationship", .stringAttributeType, false),
            ("email", .stringAttributeType, true),
            ("isDoctor", .booleanAttributeType, false),
            ("sortOrder", .integer16AttributeType, false),
            ("createdAt", .dateAttributeType, false)
        ]

        emergencyContactEntity.properties = emergencyContactAttributes.map { name, type, optional in
            let attr = NSAttributeDescription()
            attr.name = name
            attr.attributeType = type
            attr.isOptional = optional
            return attr
        }

        // UserProfile Entity
        let userProfileEntity = NSEntityDescription()
        userProfileEntity.name = "UserProfileEntity"
        userProfileEntity.managedObjectClassName = "UserProfileEntity"

        let userProfileAttributes: [(String, NSAttributeType, Bool)] = [
            ("id", .UUIDAttributeType, false),
            ("dateOfBirth", .dateAttributeType, true),
            ("gender", .stringAttributeType, true),
            ("heightCM", .floatAttributeType, false),
            ("weightKG", .floatAttributeType, false),
            ("doctorEmail", .stringAttributeType, true),
            ("createdAt", .dateAttributeType, false),
            ("updatedAt", .dateAttributeType, false)
        ]

        userProfileEntity.properties = userProfileAttributes.map { name, type, optional in
            let attr = NSAttributeDescription()
            attr.name = name
            attr.attributeType = type
            attr.isOptional = optional
            return attr
        }

        // Relationships
        let ecgToScreening = NSRelationshipDescription()
        ecgToScreening.name = "screeningResult"
        ecgToScreening.destinationEntity = screeningResultEntity
        ecgToScreening.minCount = 0
        ecgToScreening.maxCount = 1
        ecgToScreening.isOptional = true
        ecgToScreening.deleteRule = .cascadeDeleteRule

        let screeningToEcg = NSRelationshipDescription()
        screeningToEcg.name = "ecgRecord"
        screeningToEcg.destinationEntity = ecgRecordEntity
        screeningToEcg.minCount = 0
        screeningToEcg.maxCount = 1
        screeningToEcg.isOptional = true
        screeningToEcg.deleteRule = .nullifyDeleteRule

        ecgToScreening.inverseRelationship = screeningToEcg
        screeningToEcg.inverseRelationship = ecgToScreening

        ecgRecordEntity.properties.append(ecgToScreening)
        screeningResultEntity.properties.append(screeningToEcg)

        model.entities = [ecgRecordEntity, screeningResultEntity, emergencyContactEntity, userProfileEntity]

        return model
    }
}
