import Foundation

// Type alias for backward compatibility
typealias DiseaseConditions = DiseaseCondition

struct DiseaseCondition: Identifiable, Codable {
    let id: Int
    let nameEN: String
    let nameCN: String
    let category: Category

    enum Category: String, Codable, CaseIterable {
        case rhythm = "Rhythm"
        case conduction = "Conduction"
        case infarction = "Infarction"
        case hypertrophy = "Hypertrophy"
        case stTWave = "ST/T Wave"
        case pacemaker = "Pacemaker"
        case axis = "Axis"
        case other = "Other"

        var localizedName: String {
            switch self {
            case .rhythm: return String(localized: "category.rhythm")
            case .conduction: return String(localized: "category.conduction")
            case .infarction: return String(localized: "category.infarction")
            case .hypertrophy: return String(localized: "category.hypertrophy")
            case .stTWave: return String(localized: "category.stTWave")
            case .pacemaker: return String(localized: "category.pacemaker")
            case .axis: return String(localized: "category.axis")
            case .other: return String(localized: "category.other")
            }
        }
    }

    var localizedName: String {
        let languageCode = Locale.current.language.languageCode?.identifier ?? "en"
        return languageCode == "zh" ? nameCN : nameEN
    }

    // MARK: - Load All Conditions

    /// Alias for allConditions (for backward compatibility)
    static let all = allConditions

    static func loadAll() -> [DiseaseCondition] {
        return allConditions
    }

    static func condition(at index: Int) -> DiseaseCondition? {
        guard index >= 0 && index < allConditions.count else { return nil }
        return allConditions[index]
    }
}

// MARK: - All 150 Disease Conditions with Chinese Translations
extension DiseaseCondition {
    static let allConditions: [DiseaseCondition] = [
        // Rhythm (0-7)
        DiseaseCondition(id: 0, nameEN: "ABNORMAL ECG", nameCN: "异常心电图", category: .other),
        DiseaseCondition(id: 1, nameEN: "NORMAL SINUS RHYTHM", nameCN: "正常窦性心律", category: .rhythm),
        DiseaseCondition(id: 2, nameEN: "NORMAL ECG", nameCN: "正常心电图", category: .other),
        DiseaseCondition(id: 3, nameEN: "SINUS RHYTHM", nameCN: "窦性心律", category: .rhythm),
        DiseaseCondition(id: 4, nameEN: "SINUS BRADYCARDIA", nameCN: "窦性心动过缓", category: .rhythm),
        DiseaseCondition(id: 5, nameEN: "ATRIAL FIBRILLATION", nameCN: "心房颤动", category: .rhythm),
        DiseaseCondition(id: 6, nameEN: "SINUS TACHYCARDIA", nameCN: "窦性心动过速", category: .rhythm),
        DiseaseCondition(id: 7, nameEN: "otherwise normal ecg", nameCN: "其他方面正常心电图", category: .other),

        // Axis & Morphology (8-10)
        DiseaseCondition(id: 8, nameEN: "LEFT AXIS DEVIATION", nameCN: "电轴左偏", category: .axis),
        DiseaseCondition(id: 9, nameEN: "PREMATURE VENTRICULAR COMPLEXES", nameCN: "室性早搏", category: .rhythm),
        DiseaseCondition(id: 10, nameEN: "BORDERLINE ECG", nameCN: "临界心电图", category: .other),

        // Conduction (11-20)
        DiseaseCondition(id: 11, nameEN: "RIGHT BUNDLE BRANCH BLOCK", nameCN: "右束支传导阻滞", category: .conduction),
        DiseaseCondition(id: 12, nameEN: "SEPTAL INFARCT", nameCN: "间隔梗死", category: .infarction),
        DiseaseCondition(id: 13, nameEN: "LEFT ATRIAL ENLARGEMENT", nameCN: "左心房增大", category: .hypertrophy),
        DiseaseCondition(id: 14, nameEN: "NONSPECIFIC T WAVE ABNORMALITY", nameCN: "非特异性T波异常", category: .stTWave),
        DiseaseCondition(id: 15, nameEN: "LOW VOLTAGE QRS", nameCN: "低电压QRS", category: .other),
        DiseaseCondition(id: 16, nameEN: "PREMATURE ATRIAL COMPLEXES", nameCN: "房性早搏", category: .rhythm),
        DiseaseCondition(id: 17, nameEN: "ANTERIOR INFARCT", nameCN: "前壁梗死", category: .infarction),
        DiseaseCondition(id: 18, nameEN: "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK", nameCN: "不完全性右束支传导阻滞", category: .conduction),
        DiseaseCondition(id: 19, nameEN: "PREMATURE SUPRAVENTRICULAR COMPLEXES", nameCN: "室上性早搏", category: .rhythm),
        DiseaseCondition(id: 20, nameEN: "LEFT BUNDLE BRANCH BLOCK", nameCN: "左束支传导阻滞", category: .conduction),

        // ST/T Wave Changes (21-30)
        DiseaseCondition(id: 21, nameEN: "NONSPECIFIC T WAVE ABNORMALITY NOW EVIDENT IN", nameCN: "非特异性T波异常现在明显于", category: .stTWave),
        DiseaseCondition(id: 22, nameEN: "NONSPECIFIC T WAVE ABNORMALITY NO LONGER EVIDENT IN", nameCN: "非特异性T波异常不再明显于", category: .stTWave),
        DiseaseCondition(id: 23, nameEN: "T WAVE INVERSION NOW EVIDENT IN", nameCN: "T波倒置现在明显于", category: .stTWave),
        DiseaseCondition(id: 24, nameEN: "LATERAL INFARCT", nameCN: "侧壁梗死", category: .infarction),
        DiseaseCondition(id: 25, nameEN: "NONSPECIFIC ST ABNORMALITY", nameCN: "非特异性ST异常", category: .stTWave),
        DiseaseCondition(id: 26, nameEN: "LEFT VENTRICULAR HYPERTROPHY", nameCN: "左心室肥大", category: .hypertrophy),
        DiseaseCondition(id: 27, nameEN: "T WAVE INVERSION NO LONGER EVIDENT IN", nameCN: "T波倒置不再明显于", category: .stTWave),
        DiseaseCondition(id: 28, nameEN: "WITH RAPID VENTRICULAR RESPONSE", nameCN: "伴快速心室率", category: .rhythm),
        DiseaseCondition(id: 29, nameEN: "QT HAS SHORTENED", nameCN: "QT间期缩短", category: .other),
        DiseaseCondition(id: 30, nameEN: "QT HAS LENGTHENED", nameCN: "QT间期延长", category: .other),

        // More Rhythm (31-40)
        DiseaseCondition(id: 31, nameEN: "FUSION COMPLEXES", nameCN: "融合波", category: .rhythm),
        DiseaseCondition(id: 32, nameEN: "ATRIAL FLUTTER", nameCN: "心房扑动", category: .rhythm),
        DiseaseCondition(id: 33, nameEN: "MARKED SINUS BRADYCARDIA", nameCN: "显著窦性心动过缓", category: .rhythm),
        DiseaseCondition(id: 34, nameEN: "WITH SINUS ARRHYTHMIA", nameCN: "伴窦性心律不齐", category: .rhythm),
        DiseaseCondition(id: 35, nameEN: "NONSPECIFIC ST AND T WAVE ABNORMALITY", nameCN: "非特异性ST和T波异常", category: .stTWave),
        DiseaseCondition(id: 36, nameEN: "LEFT ANTERIOR FASCICULAR BLOCK", nameCN: "左前分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 37, nameEN: "RIGHT AXIS DEVIATION", nameCN: "电轴右偏", category: .axis),
        DiseaseCondition(id: 38, nameEN: "ECTOPIC ATRIAL RHYTHM", nameCN: "异位房性心律", category: .rhythm),
        DiseaseCondition(id: 39, nameEN: "UNDETERMINED RHYTHM", nameCN: "不确定心律", category: .rhythm),
        DiseaseCondition(id: 40, nameEN: "ANTEROSEPTAL INFARCT", nameCN: "前间壁梗死", category: .infarction),

        // More conditions (41-60)
        DiseaseCondition(id: 41, nameEN: "RIGHTWARD AXIS", nameCN: "电轴右移", category: .axis),
        DiseaseCondition(id: 42, nameEN: "ST NOW DEPRESSED IN", nameCN: "ST段压低于", category: .stTWave),
        DiseaseCondition(id: 43, nameEN: "WITH SHORT PR", nameCN: "伴短PR间期", category: .conduction),
        DiseaseCondition(id: 44, nameEN: "WITH MARKED SINUS ARRHYTHMIA", nameCN: "伴显著窦性心律不齐", category: .rhythm),
        DiseaseCondition(id: 45, nameEN: "ST NO LONGER DEPRESSED IN", nameCN: "ST段不再压低于", category: .stTWave),
        DiseaseCondition(id: 46, nameEN: "INVERTED T WAVES HAVE REPLACED NONSPECIFIC T WAVE ABNORMALITY IN", nameCN: "T波倒置已取代非特异性T波异常于", category: .stTWave),
        DiseaseCondition(id: 47, nameEN: "NON-SPECIFIC CHANGE IN ST SEGMENT IN", nameCN: "非特异性ST段改变于", category: .stTWave),
        DiseaseCondition(id: 48, nameEN: "NONSPECIFIC T WAVE ABNORMALITY HAS REPLACED INVERTED T WAVES IN", nameCN: "非特异性T波异常已取代T波倒置于", category: .stTWave),
        DiseaseCondition(id: 49, nameEN: "JUNCTIONAL RHYTHM", nameCN: "交界性心律", category: .rhythm),
        DiseaseCondition(id: 50, nameEN: "ELECTRONIC ATRIAL PACEMAKER", nameCN: "电子心房起搏器", category: .pacemaker),

        // Pacemaker & More (51-70)
        DiseaseCondition(id: 51, nameEN: "ABERRANT CONDUCTION", nameCN: "差异性传导", category: .conduction),
        DiseaseCondition(id: 52, nameEN: "ELECTRONIC VENTRICULAR PACEMAKER", nameCN: "电子心室起搏器", category: .pacemaker),
        DiseaseCondition(id: 53, nameEN: "T WAVE INVERSION LESS EVIDENT IN", nameCN: "T波倒置减轻于", category: .stTWave),
        DiseaseCondition(id: 54, nameEN: "ANTEROLATERAL INFARCT", nameCN: "前侧壁梗死", category: .infarction),
        DiseaseCondition(id: 55, nameEN: "WITH REPOLARIZATION ABNORMALITY", nameCN: "伴复极异常", category: .stTWave),
        DiseaseCondition(id: 56, nameEN: "RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY", nameCN: "V1导联RSR'或QR图形提示右室传导延迟", category: .conduction),
        DiseaseCondition(id: 57, nameEN: "T WAVE INVERSION MORE EVIDENT IN", nameCN: "T波倒置加重于", category: .stTWave),
        DiseaseCondition(id: 58, nameEN: "WIDE QRS RHYTHM", nameCN: "宽QRS心律", category: .rhythm),
        DiseaseCondition(id: 59, nameEN: "WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES", nameCN: "伴室性早搏或差异性传导", category: .rhythm),
        DiseaseCondition(id: 60, nameEN: "RIGHT ATRIAL ENLARGEMENT", nameCN: "右心房增大", category: .hypertrophy),

        // More conditions (61-80)
        DiseaseCondition(id: 61, nameEN: "INFERIOR INFARCT", nameCN: "下壁梗死", category: .infarction),
        DiseaseCondition(id: 62, nameEN: "INCOMPLETE LEFT BUNDLE BRANCH BLOCK", nameCN: "不完全性左束支传导阻滞", category: .conduction),
        DiseaseCondition(id: 63, nameEN: "VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY", nameCN: "左心室肥大电压标准", category: .hypertrophy),
        DiseaseCondition(id: 64, nameEN: "OR DIGITALIS EFFECT", nameCN: "或洋地黄效应", category: .other),
        DiseaseCondition(id: 65, nameEN: "BIFASCICULAR BLOCK", nameCN: "双分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 66, nameEN: "ST NO LONGER ELEVATED IN", nameCN: "ST段不再抬高于", category: .stTWave),
        DiseaseCondition(id: 67, nameEN: "WITH SLOW VENTRICULAR RESPONSE", nameCN: "伴缓慢心室率", category: .rhythm),
        DiseaseCondition(id: 68, nameEN: "ST ELEVATION NOW PRESENT IN", nameCN: "ST段抬高现于", category: .stTWave),
        DiseaseCondition(id: 69, nameEN: "PREMATURE ECTOPIC COMPLEXES", nameCN: "异位早搏", category: .rhythm),
        DiseaseCondition(id: 70, nameEN: "LEFT POSTERIOR FASCICULAR BLOCK", nameCN: "左后分支传导阻滞", category: .conduction),

        // More conditions (71-90)
        DiseaseCondition(id: 71, nameEN: "T WAVE AMPLITUDE HAS DECREASED IN", nameCN: "T波振幅降低于", category: .stTWave),
        DiseaseCondition(id: 72, nameEN: "WITH A COMPETING JUNCTIONAL PACEMAKER", nameCN: "伴竞争性交界性起搏点", category: .rhythm),
        DiseaseCondition(id: 73, nameEN: "RIGHT SUPERIOR AXIS DEVIATION", nameCN: "电轴右上偏", category: .axis),
        DiseaseCondition(id: 74, nameEN: "BIATRIAL ENLARGEMENT", nameCN: "双心房增大", category: .hypertrophy),
        DiseaseCondition(id: 75, nameEN: "VENTRICULAR-PACED RHYTHM", nameCN: "心室起搏心律", category: .pacemaker),
        DiseaseCondition(id: 76, nameEN: "ATRIAL-PACED RHYTHM", nameCN: "心房起搏心律", category: .pacemaker),
        DiseaseCondition(id: 77, nameEN: "T WAVE AMPLITUDE HAS INCREASED IN", nameCN: "T波振幅增高于", category: .stTWave),
        DiseaseCondition(id: 78, nameEN: "WITH QRS WIDENING", nameCN: "伴QRS增宽", category: .conduction),
        DiseaseCondition(id: 79, nameEN: "WITH 1ST DEGREE AV BLOCK", nameCN: "伴一度房室传导阻滞", category: .conduction),
        DiseaseCondition(id: 80, nameEN: "PROLONGED QT", nameCN: "QT间期延长", category: .other),

        // More conditions (81-100)
        DiseaseCondition(id: 81, nameEN: "WITH PROLONGED AV CONDUCTION", nameCN: "伴房室传导延迟", category: .conduction),
        DiseaseCondition(id: 82, nameEN: "RIGHT VENTRICULAR HYPERTROPHY", nameCN: "右心室肥大", category: .hypertrophy),
        DiseaseCondition(id: 83, nameEN: "WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY", nameCN: "伴QRS增宽和复极异常", category: .conduction),
        DiseaseCondition(id: 84, nameEN: "ATRIAL-SENSED VENTRICULAR-PACED RHYTHM", nameCN: "心房感知心室起搏心律", category: .pacemaker),
        DiseaseCondition(id: 85, nameEN: "AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER", nameCN: "房室顺序或双腔电子起搏器", category: .pacemaker),
        DiseaseCondition(id: 86, nameEN: "PULMONARY DISEASE PATTERN", nameCN: "肺部疾病图形", category: .other),
        DiseaseCondition(id: 87, nameEN: "ACUTE MI / STEMI", nameCN: "急性心肌梗死/ST段抬高型心肌梗死", category: .infarction),
        DiseaseCondition(id: 88, nameEN: "INFERIOR-POSTERIOR INFARCT", nameCN: "下后壁梗死", category: .infarction),
        DiseaseCondition(id: 89, nameEN: "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY", nameCN: "非特异性室内传导延迟", category: .conduction),
        DiseaseCondition(id: 90, nameEN: "PREMATURE VENTRICULAR AND FUSION COMPLEXES", nameCN: "室性早搏和融合波", category: .rhythm),

        // More conditions (91-110)
        DiseaseCondition(id: 91, nameEN: "IN A PATTERN OF BIGEMINY", nameCN: "呈二联律形式", category: .rhythm),
        DiseaseCondition(id: 92, nameEN: "AV DUAL-PACED RHYTHM", nameCN: "房室双腔起搏心律", category: .pacemaker),
        DiseaseCondition(id: 93, nameEN: "SUPRAVENTRICULAR TACHYCARDIA", nameCN: "室上性心动过速", category: .rhythm),
        DiseaseCondition(id: 94, nameEN: "VENTRICULAR-PACED COMPLEXES", nameCN: "心室起搏波群", category: .pacemaker),
        DiseaseCondition(id: 95, nameEN: "WIDE QRS TACHYCARDIA", nameCN: "宽QRS心动过速", category: .rhythm),
        DiseaseCondition(id: 96, nameEN: "RSR' PATTERN IN V1", nameCN: "V1导联RSR'图形", category: .conduction),
        DiseaseCondition(id: 97, nameEN: "ST LESS DEPRESSED IN", nameCN: "ST段压低减轻于", category: .stTWave),
        DiseaseCondition(id: 98, nameEN: "VENTRICULAR TACHYCARDIA", nameCN: "室性心动过速", category: .rhythm),
        DiseaseCondition(id: 99, nameEN: "EARLY REPOLARIZATION", nameCN: "早期复极", category: .stTWave),
        DiseaseCondition(id: 100, nameEN: "ST MORE DEPRESSED IN", nameCN: "ST段压低加重于", category: .stTWave),

        // More conditions (101-120)
        DiseaseCondition(id: 101, nameEN: "ANTEROLATERAL LEADS", nameCN: "前侧壁导联", category: .other),
        DiseaseCondition(id: 102, nameEN: "ELECTRONIC DEMAND PACING", nameCN: "电子按需起搏", category: .pacemaker),
        DiseaseCondition(id: 103, nameEN: "RBBB AND LEFT ANTERIOR FASCICULAR BLOCK", nameCN: "右束支传导阻滞伴左前分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 104, nameEN: "LATERAL INJURY PATTERN", nameCN: "侧壁损伤图形", category: .infarction),
        DiseaseCondition(id: 105, nameEN: "BIVENTRICULAR PACEMAKER DETECTED", nameCN: "检测到双心室起搏器", category: .pacemaker),
        DiseaseCondition(id: 106, nameEN: "SUSPECT UNSPECIFIED PACEMAKER FAILURE", nameCN: "疑似起搏器故障", category: .pacemaker),
        DiseaseCondition(id: 107, nameEN: "WOLFF-PARKINSON-WHITE", nameCN: "预激综合征(WPW)", category: .conduction),
        DiseaseCondition(id: 108, nameEN: "WITH VENTRICULAR ESCAPE COMPLEXES", nameCN: "伴室性逸搏", category: .rhythm),
        DiseaseCondition(id: 109, nameEN: "INFERIOR INJURY PATTERN", nameCN: "下壁损伤图形", category: .infarction),
        DiseaseCondition(id: 110, nameEN: "CONSIDER RIGHT VENTRICULAR INVOLVEMENT IN ACUTE INFERIOR INFARCT", nameCN: "急性下壁梗死考虑右室受累", category: .infarction),

        // More conditions (111-130)
        DiseaseCondition(id: 111, nameEN: "ST ELEVATION HAS REPLACED ST DEPRESSION IN", nameCN: "ST段抬高已取代ST段压低于", category: .stTWave),
        DiseaseCondition(id: 112, nameEN: "NONSPECIFIC INTRAVENTRICULAR BLOCK", nameCN: "非特异性室内传导阻滞", category: .conduction),
        DiseaseCondition(id: 113, nameEN: "MASKED BY FASCICULAR BLOCK", nameCN: "被分支阻滞掩盖", category: .conduction),
        DiseaseCondition(id: 114, nameEN: "PEDIATRIC ECG ANALYSIS", nameCN: "儿童心电图分析", category: .other),
        DiseaseCondition(id: 115, nameEN: "BLOCKED", nameCN: "传导阻滞", category: .conduction),
        DiseaseCondition(id: 116, nameEN: "WITH UNDETERMINED RHYTHM IRREGULARITY", nameCN: "伴不确定心律不齐", category: .rhythm),
        DiseaseCondition(id: 117, nameEN: "LEFTWARD AXIS", nameCN: "电轴左移", category: .axis),
        DiseaseCondition(id: 118, nameEN: "WITH 2ND DEGREE SA BLOCK MOBITZ I", nameCN: "伴二度I型窦房传导阻滞", category: .conduction),
        DiseaseCondition(id: 119, nameEN: "ACUTE", nameCN: "急性", category: .other),
        DiseaseCondition(id: 120, nameEN: "ABNORMAL LEFT AXIS DEVIATION", nameCN: "异常电轴左偏", category: .axis),

        // Final conditions (121-149)
        DiseaseCondition(id: 121, nameEN: "WITH COMPLETE HEART BLOCK", nameCN: "伴完全性心脏传导阻滞", category: .conduction),
        DiseaseCondition(id: 122, nameEN: "NO P-WAVES FOUND", nameCN: "未发现P波", category: .rhythm),
        DiseaseCondition(id: 123, nameEN: "ST LESS ELEVATED IN", nameCN: "ST段抬高减轻于", category: .stTWave),
        DiseaseCondition(id: 124, nameEN: "WITH RETROGRADE CONDUCTION", nameCN: "伴逆向传导", category: .conduction),
        DiseaseCondition(id: 125, nameEN: "ST MORE ELEVATED IN", nameCN: "ST段抬高加重于", category: .stTWave),
        DiseaseCondition(id: 126, nameEN: "JUNCTIONAL BRADYCARDIA", nameCN: "交界性心动过缓", category: .rhythm),
        DiseaseCondition(id: 127, nameEN: "WITH VARIABLE AV BLOCK", nameCN: "伴可变房室传导阻滞", category: .conduction),
        DiseaseCondition(id: 128, nameEN: "ANTERIOR INJURY PATTERN", nameCN: "前壁损伤图形", category: .infarction),
        DiseaseCondition(id: 129, nameEN: "WITH JUNCTIONAL ESCAPE COMPLEXES", nameCN: "伴交界性逸搏", category: .rhythm),
        DiseaseCondition(id: 130, nameEN: "ACUTE MI", nameCN: "急性心肌梗死", category: .infarction),
        DiseaseCondition(id: 131, nameEN: "ACUTE PERICARDITIS", nameCN: "急性心包炎", category: .other),
        DiseaseCondition(id: 132, nameEN: "POSTERIOR INFARCT", nameCN: "后壁梗死", category: .infarction),
        DiseaseCondition(id: 133, nameEN: "IDIOVENTRICULAR RHYTHM", nameCN: "室性自主心律", category: .rhythm),
        DiseaseCondition(id: 134, nameEN: "WITH 2ND DEGREE SA BLOCK MOBITZ II", nameCN: "伴二度II型窦房传导阻滞", category: .conduction),
        DiseaseCondition(id: 135, nameEN: "R IN AVL", nameCN: "aVL导联R波", category: .other),
        DiseaseCondition(id: 136, nameEN: "SINUS/ATRIAL CAPTURE", nameCN: "窦性/房性夺获", category: .rhythm),
        DiseaseCondition(id: 137, nameEN: "AV DUAL-PACED COMPLEXES", nameCN: "房室双腔起搏波群", category: .pacemaker),
        DiseaseCondition(id: 138, nameEN: "INFEROLATERAL INJURY PATTERN", nameCN: "下侧壁损伤图形", category: .infarction),
        DiseaseCondition(id: 139, nameEN: "RBBB AND LEFT POSTERIOR FASCICULAR BLOCK", nameCN: "右束支传导阻滞伴左后分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 140, nameEN: "ANTEROLATERAL INJURY PATTERN", nameCN: "前侧壁损伤图形", category: .infarction),
        DiseaseCondition(id: 141, nameEN: "ATRIAL-PACED COMPLEXES", nameCN: "心房起搏波群", category: .pacemaker),
        DiseaseCondition(id: 142, nameEN: "WITH SINUS PAUSE", nameCN: "伴窦性停搏", category: .rhythm),
        DiseaseCondition(id: 143, nameEN: "BIVENTRICULAR HYPERTROPHY", nameCN: "双心室肥大", category: .hypertrophy),
        DiseaseCondition(id: 144, nameEN: "ABNORMAL RIGHT AXIS DEVIATION", nameCN: "异常电轴右偏", category: .axis),
        DiseaseCondition(id: 145, nameEN: "SUPRAVENTRICULAR COMPLEXES", nameCN: "室上性波群", category: .rhythm),
        DiseaseCondition(id: 146, nameEN: "WITH 2ND DEGREE AV BLOCK MOBITZ I", nameCN: "伴二度I型房室传导阻滞", category: .conduction),
        DiseaseCondition(id: 147, nameEN: "WITH 2:1 AV CONDUCTION", nameCN: "伴2:1房室传导", category: .conduction),
        DiseaseCondition(id: 148, nameEN: "WITH AV DISSOCIATION", nameCN: "伴房室分离", category: .conduction),
        DiseaseCondition(id: 149, nameEN: "MULTIFOCAL ATRIAL TACHYCARDIA", nameCN: "多源性房性心动过速", category: .rhythm),
    ]
}
