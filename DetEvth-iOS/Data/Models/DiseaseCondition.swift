import Foundation

// Type alias for backward compatibility
typealias DiseaseConditions = DiseaseCondition

struct DiseaseCondition: Identifiable, Codable {
    let id: Int
    let nameEN: String
    let nameCN: String
    let category: Category

    enum Category: String, Codable, CaseIterable {
        case rhythm = "rhythm"
        case conduction = "conduction"
        case infarction = "infarction"
        case hypertrophy = "hypertrophy"
        case stTWave = "st/t wave"
        case pacemaker = "pacemaker"
        case axis = "axis"
        case other = "other"

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
        DiseaseCondition(id: 0, nameEN: "abnormal ecg", nameCN: "异常心电图", category: .other),
        DiseaseCondition(id: 1, nameEN: "normal sinus rhythm", nameCN: "正常窦性心律", category: .rhythm),
        DiseaseCondition(id: 2, nameEN: "normal ecg", nameCN: "正常心电图", category: .other),
        DiseaseCondition(id: 3, nameEN: "sinus rhythm", nameCN: "窦性心律", category: .rhythm),
        DiseaseCondition(id: 4, nameEN: "sinus bradycardia", nameCN: "窦性心动过缓", category: .rhythm),
        DiseaseCondition(id: 5, nameEN: "atrial fibrillation", nameCN: "心房颤动", category: .rhythm),
        DiseaseCondition(id: 6, nameEN: "sinus tachycardia", nameCN: "窦性心动过速", category: .rhythm),
        DiseaseCondition(id: 7, nameEN: "otherwise normal ecg", nameCN: "其他方面正常心电图", category: .other),

        // Axis & Morphology (8-10)
        DiseaseCondition(id: 8, nameEN: "left axis deviation", nameCN: "电轴左偏", category: .axis),
        DiseaseCondition(id: 9, nameEN: "premature ventricular complexes", nameCN: "室性早搏", category: .rhythm),
        DiseaseCondition(id: 10, nameEN: "borderline ecg", nameCN: "临界心电图", category: .other),

        // Conduction (11-20)
        DiseaseCondition(id: 11, nameEN: "right bundle branch block", nameCN: "右束支传导阻滞", category: .conduction),
        DiseaseCondition(id: 12, nameEN: "septal infarct", nameCN: "间隔梗死", category: .infarction),
        DiseaseCondition(id: 13, nameEN: "left atrial enlargement", nameCN: "左心房增大", category: .hypertrophy),
        DiseaseCondition(id: 14, nameEN: "nonspecific t wave abnormality", nameCN: "非特异性T波异常", category: .stTWave),
        DiseaseCondition(id: 15, nameEN: "low voltage qrs", nameCN: "低电压QRS", category: .other),
        DiseaseCondition(id: 16, nameEN: "premature atrial complexes", nameCN: "房性早搏", category: .rhythm),
        DiseaseCondition(id: 17, nameEN: "anterior infarct", nameCN: "前壁梗死", category: .infarction),
        DiseaseCondition(id: 18, nameEN: "incomplete right bundle branch block", nameCN: "不完全性右束支传导阻滞", category: .conduction),
        DiseaseCondition(id: 19, nameEN: "premature supraventricular complexes", nameCN: "室上性早搏", category: .rhythm),
        DiseaseCondition(id: 20, nameEN: "left bundle branch block", nameCN: "左束支传导阻滞", category: .conduction),

        // ST/T Wave Changes (21-30)
        DiseaseCondition(id: 21, nameEN: "nonspecific t wave abnormality now evident in", nameCN: "非特异性T波异常现在明显于", category: .stTWave),
        DiseaseCondition(id: 22, nameEN: "nonspecific t wave abnormality no longer evident in", nameCN: "非特异性T波异常不再明显于", category: .stTWave),
        DiseaseCondition(id: 23, nameEN: "t wave inversion now evident in", nameCN: "T波倒置现在明显于", category: .stTWave),
        DiseaseCondition(id: 24, nameEN: "lateral infarct", nameCN: "侧壁梗死", category: .infarction),
        DiseaseCondition(id: 25, nameEN: "nonspecific st abnormality", nameCN: "非特异性ST异常", category: .stTWave),
        DiseaseCondition(id: 26, nameEN: "left ventricular hypertrophy", nameCN: "左心室肥大", category: .hypertrophy),
        DiseaseCondition(id: 27, nameEN: "t wave inversion no longer evident in", nameCN: "T波倒置不再明显于", category: .stTWave),
        DiseaseCondition(id: 28, nameEN: "with rapid ventricular response", nameCN: "伴快速心室率", category: .rhythm),
        DiseaseCondition(id: 29, nameEN: "qt has shortened", nameCN: "QT间期缩短", category: .other),
        DiseaseCondition(id: 30, nameEN: "qt has lengthened", nameCN: "QT间期延长", category: .other),

        // More Rhythm (31-40)
        DiseaseCondition(id: 31, nameEN: "fusion complexes", nameCN: "融合波", category: .rhythm),
        DiseaseCondition(id: 32, nameEN: "atrial flutter", nameCN: "心房扑动", category: .rhythm),
        DiseaseCondition(id: 33, nameEN: "marked sinus bradycardia", nameCN: "显著窦性心动过缓", category: .rhythm),
        DiseaseCondition(id: 34, nameEN: "with sinus arrhythmia", nameCN: "伴窦性心律不齐", category: .rhythm),
        DiseaseCondition(id: 35, nameEN: "nonspecific st and t wave abnormality", nameCN: "非特异性ST和T波异常", category: .stTWave),
        DiseaseCondition(id: 36, nameEN: "left anterior fascicular block", nameCN: "左前分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 37, nameEN: "right axis deviation", nameCN: "电轴右偏", category: .axis),
        DiseaseCondition(id: 38, nameEN: "ectopic atrial rhythm", nameCN: "异位房性心律", category: .rhythm),
        DiseaseCondition(id: 39, nameEN: "undetermined rhythm", nameCN: "不确定心律", category: .rhythm),
        DiseaseCondition(id: 40, nameEN: "anteroseptal infarct", nameCN: "前间壁梗死", category: .infarction),

        // More conditions (41-60)
        DiseaseCondition(id: 41, nameEN: "rightward axis", nameCN: "电轴右移", category: .axis),
        DiseaseCondition(id: 42, nameEN: "st now depressed in", nameCN: "ST段压低于", category: .stTWave),
        DiseaseCondition(id: 43, nameEN: "with short pr", nameCN: "伴短PR间期", category: .conduction),
        DiseaseCondition(id: 44, nameEN: "with marked sinus arrhythmia", nameCN: "伴显著窦性心律不齐", category: .rhythm),
        DiseaseCondition(id: 45, nameEN: "st no longer depressed in", nameCN: "ST段不再压低于", category: .stTWave),
        DiseaseCondition(id: 46, nameEN: "inverted t waves have replaced nonspecific t wave abnormality in", nameCN: "T波倒置已取代非特异性T波异常于", category: .stTWave),
        DiseaseCondition(id: 47, nameEN: "non-specific change in st segment in", nameCN: "非特异性ST段改变于", category: .stTWave),
        DiseaseCondition(id: 48, nameEN: "nonspecific t wave abnormality has replaced inverted t waves in", nameCN: "非特异性T波异常已取代T波倒置于", category: .stTWave),
        DiseaseCondition(id: 49, nameEN: "junctional rhythm", nameCN: "交界性心律", category: .rhythm),
        DiseaseCondition(id: 50, nameEN: "electronic atrial pacemaker", nameCN: "电子心房起搏器", category: .pacemaker),

        // Pacemaker & More (51-70)
        DiseaseCondition(id: 51, nameEN: "aberrant conduction", nameCN: "差异性传导", category: .conduction),
        DiseaseCondition(id: 52, nameEN: "electronic ventricular pacemaker", nameCN: "电子心室起搏器", category: .pacemaker),
        DiseaseCondition(id: 53, nameEN: "t wave inversion less evident in", nameCN: "T波倒置减轻于", category: .stTWave),
        DiseaseCondition(id: 54, nameEN: "anterolateral infarct", nameCN: "前侧壁梗死", category: .infarction),
        DiseaseCondition(id: 55, nameEN: "with repolarization abnormality", nameCN: "伴复极异常", category: .stTWave),
        DiseaseCondition(id: 56, nameEN: "rsr' or qr pattern in v1 suggests right ventricular conduction delay", nameCN: "V1导联RSR'或QR图形提示右室传导延迟", category: .conduction),
        DiseaseCondition(id: 57, nameEN: "t wave inversion more evident in", nameCN: "T波倒置加重于", category: .stTWave),
        DiseaseCondition(id: 58, nameEN: "wide qrs rhythm", nameCN: "宽QRS心律", category: .rhythm),
        DiseaseCondition(id: 59, nameEN: "with premature ventricular or aberrantly conducted complexes", nameCN: "伴室性早搏或差异性传导", category: .rhythm),
        DiseaseCondition(id: 60, nameEN: "right atrial enlargement", nameCN: "右心房增大", category: .hypertrophy),

        // More conditions (61-80)
        DiseaseCondition(id: 61, nameEN: "inferior infarct", nameCN: "下壁梗死", category: .infarction),
        DiseaseCondition(id: 62, nameEN: "incomplete left bundle branch block", nameCN: "不完全性左束支传导阻滞", category: .conduction),
        DiseaseCondition(id: 63, nameEN: "voltage criteria for left ventricular hypertrophy", nameCN: "左心室肥大电压标准", category: .hypertrophy),
        DiseaseCondition(id: 64, nameEN: "or digitalis effect", nameCN: "或洋地黄效应", category: .other),
        DiseaseCondition(id: 65, nameEN: "bifascicular block", nameCN: "双分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 66, nameEN: "st no longer elevated in", nameCN: "ST段不再抬高于", category: .stTWave),
        DiseaseCondition(id: 67, nameEN: "with slow ventricular response", nameCN: "伴缓慢心室率", category: .rhythm),
        DiseaseCondition(id: 68, nameEN: "st elevation now present in", nameCN: "ST段抬高现于", category: .stTWave),
        DiseaseCondition(id: 69, nameEN: "premature ectopic complexes", nameCN: "异位早搏", category: .rhythm),
        DiseaseCondition(id: 70, nameEN: "left posterior fascicular block", nameCN: "左后分支传导阻滞", category: .conduction),

        // More conditions (71-90)
        DiseaseCondition(id: 71, nameEN: "t wave amplitude has decreased in", nameCN: "T波振幅降低于", category: .stTWave),
        DiseaseCondition(id: 72, nameEN: "with a competing junctional pacemaker", nameCN: "伴竞争性交界性起搏点", category: .rhythm),
        DiseaseCondition(id: 73, nameEN: "right superior axis deviation", nameCN: "电轴右上偏", category: .axis),
        DiseaseCondition(id: 74, nameEN: "biatrial enlargement", nameCN: "双心房增大", category: .hypertrophy),
        DiseaseCondition(id: 75, nameEN: "ventricular-paced rhythm", nameCN: "心室起搏心律", category: .pacemaker),
        DiseaseCondition(id: 76, nameEN: "atrial-paced rhythm", nameCN: "心房起搏心律", category: .pacemaker),
        DiseaseCondition(id: 77, nameEN: "t wave amplitude has increased in", nameCN: "T波振幅增高于", category: .stTWave),
        DiseaseCondition(id: 78, nameEN: "with qrs widening", nameCN: "伴QRS增宽", category: .conduction),
        DiseaseCondition(id: 79, nameEN: "with 1st degree av block", nameCN: "伴一度房室传导阻滞", category: .conduction),
        DiseaseCondition(id: 80, nameEN: "prolonged qt", nameCN: "QT间期延长", category: .other),

        // More conditions (81-100)
        DiseaseCondition(id: 81, nameEN: "with prolonged av conduction", nameCN: "伴房室传导延迟", category: .conduction),
        DiseaseCondition(id: 82, nameEN: "right ventricular hypertrophy", nameCN: "右心室肥大", category: .hypertrophy),
        DiseaseCondition(id: 83, nameEN: "with qrs widening and repolarization abnormality", nameCN: "伴QRS增宽和复极异常", category: .conduction),
        DiseaseCondition(id: 84, nameEN: "atrial-sensed ventricular-paced rhythm", nameCN: "心房感知心室起搏心律", category: .pacemaker),
        DiseaseCondition(id: 85, nameEN: "av sequential or dual chamber electronic pacemaker", nameCN: "房室顺序或双腔电子起搏器", category: .pacemaker),
        DiseaseCondition(id: 86, nameEN: "pulmonary disease pattern", nameCN: "肺部疾病图形", category: .other),
        DiseaseCondition(id: 87, nameEN: "acute mi / stemi", nameCN: "急性心肌梗死/ST段抬高型心肌梗死", category: .infarction),
        DiseaseCondition(id: 88, nameEN: "inferior-posterior infarct", nameCN: "下后壁梗死", category: .infarction),
        DiseaseCondition(id: 89, nameEN: "nonspecific intraventricular conduction delay", nameCN: "非特异性室内传导延迟", category: .conduction),
        DiseaseCondition(id: 90, nameEN: "premature ventricular and fusion complexes", nameCN: "室性早搏和融合波", category: .rhythm),

        // More conditions (91-110)
        DiseaseCondition(id: 91, nameEN: "in a pattern of bigeminy", nameCN: "呈二联律形式", category: .rhythm),
        DiseaseCondition(id: 92, nameEN: "av dual-paced rhythm", nameCN: "房室双腔起搏心律", category: .pacemaker),
        DiseaseCondition(id: 93, nameEN: "supraventricular tachycardia", nameCN: "室上性心动过速", category: .rhythm),
        DiseaseCondition(id: 94, nameEN: "ventricular-paced complexes", nameCN: "心室起搏波群", category: .pacemaker),
        DiseaseCondition(id: 95, nameEN: "wide qrs tachycardia", nameCN: "宽QRS心动过速", category: .rhythm),
        DiseaseCondition(id: 96, nameEN: "rsr' pattern in v1", nameCN: "V1导联RSR'图形", category: .conduction),
        DiseaseCondition(id: 97, nameEN: "st less depressed in", nameCN: "ST段压低减轻于", category: .stTWave),
        DiseaseCondition(id: 98, nameEN: "ventricular tachycardia", nameCN: "室性心动过速", category: .rhythm),
        DiseaseCondition(id: 99, nameEN: "early repolarization", nameCN: "早期复极", category: .stTWave),
        DiseaseCondition(id: 100, nameEN: "st more depressed in", nameCN: "ST段压低加重于", category: .stTWave),

        // More conditions (101-120)
        DiseaseCondition(id: 101, nameEN: "anterolateral leads", nameCN: "前侧壁导联", category: .other),
        DiseaseCondition(id: 102, nameEN: "electronic demand pacing", nameCN: "电子按需起搏", category: .pacemaker),
        DiseaseCondition(id: 103, nameEN: "rbbb and left anterior fascicular block", nameCN: "右束支传导阻滞伴左前分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 104, nameEN: "lateral injury pattern", nameCN: "侧壁损伤图形", category: .infarction),
        DiseaseCondition(id: 105, nameEN: "biventricular pacemaker detected", nameCN: "检测到双心室起搏器", category: .pacemaker),
        DiseaseCondition(id: 106, nameEN: "suspect unspecified pacemaker failure", nameCN: "疑似起搏器故障", category: .pacemaker),
        DiseaseCondition(id: 107, nameEN: "wolff-parkinson-white", nameCN: "预激综合征(WPW)", category: .conduction),
        DiseaseCondition(id: 108, nameEN: "with ventricular escape complexes", nameCN: "伴室性逸搏", category: .rhythm),
        DiseaseCondition(id: 109, nameEN: "inferior injury pattern", nameCN: "下壁损伤图形", category: .infarction),
        DiseaseCondition(id: 110, nameEN: "consider right ventricular involvement in acute inferior infarct", nameCN: "急性下壁梗死考虑右室受累", category: .infarction),

        // More conditions (111-130)
        DiseaseCondition(id: 111, nameEN: "st elevation has replaced st depression in", nameCN: "ST段抬高已取代ST段压低于", category: .stTWave),
        DiseaseCondition(id: 112, nameEN: "nonspecific intraventricular block", nameCN: "非特异性室内传导阻滞", category: .conduction),
        DiseaseCondition(id: 113, nameEN: "masked by fascicular block", nameCN: "被分支阻滞掩盖", category: .conduction),
        DiseaseCondition(id: 114, nameEN: "pediatric ecg analysis", nameCN: "儿童心电图分析", category: .other),
        DiseaseCondition(id: 115, nameEN: "blocked", nameCN: "传导阻滞", category: .conduction),
        DiseaseCondition(id: 116, nameEN: "with undetermined rhythm irregularity", nameCN: "伴不确定心律不齐", category: .rhythm),
        DiseaseCondition(id: 117, nameEN: "leftward axis", nameCN: "电轴左移", category: .axis),
        DiseaseCondition(id: 118, nameEN: "with 2nd degree sa block mobitz i", nameCN: "伴二度I型窦房传导阻滞", category: .conduction),
        DiseaseCondition(id: 119, nameEN: "acute", nameCN: "急性", category: .other),
        DiseaseCondition(id: 120, nameEN: "abnormal left axis deviation", nameCN: "异常电轴左偏", category: .axis),

        // Final conditions (121-149)
        DiseaseCondition(id: 121, nameEN: "with complete heart block", nameCN: "伴完全性心脏传导阻滞", category: .conduction),
        DiseaseCondition(id: 122, nameEN: "no p-waves found", nameCN: "未发现P波", category: .rhythm),
        DiseaseCondition(id: 123, nameEN: "st less elevated in", nameCN: "ST段抬高减轻于", category: .stTWave),
        DiseaseCondition(id: 124, nameEN: "with retrograde conduction", nameCN: "伴逆向传导", category: .conduction),
        DiseaseCondition(id: 125, nameEN: "st more elevated in", nameCN: "ST段抬高加重于", category: .stTWave),
        DiseaseCondition(id: 126, nameEN: "junctional bradycardia", nameCN: "交界性心动过缓", category: .rhythm),
        DiseaseCondition(id: 127, nameEN: "with variable av block", nameCN: "伴可变房室传导阻滞", category: .conduction),
        DiseaseCondition(id: 128, nameEN: "anterior injury pattern", nameCN: "前壁损伤图形", category: .infarction),
        DiseaseCondition(id: 129, nameEN: "with junctional escape complexes", nameCN: "伴交界性逸搏", category: .rhythm),
        DiseaseCondition(id: 130, nameEN: "acute mi", nameCN: "急性心肌梗死", category: .infarction),
        DiseaseCondition(id: 131, nameEN: "acute pericarditis", nameCN: "急性心包炎", category: .other),
        DiseaseCondition(id: 132, nameEN: "posterior infarct", nameCN: "后壁梗死", category: .infarction),
        DiseaseCondition(id: 133, nameEN: "idioventricular rhythm", nameCN: "室性自主心律", category: .rhythm),
        DiseaseCondition(id: 134, nameEN: "with 2nd degree sa block mobitz ii", nameCN: "伴二度II型窦房传导阻滞", category: .conduction),
        DiseaseCondition(id: 135, nameEN: "r in avl", nameCN: "aVL导联R波", category: .other),
        DiseaseCondition(id: 136, nameEN: "sinus/atrial capture", nameCN: "窦性/房性夺获", category: .rhythm),
        DiseaseCondition(id: 137, nameEN: "av dual-paced complexes", nameCN: "房室双腔起搏波群", category: .pacemaker),
        DiseaseCondition(id: 138, nameEN: "inferolateral injury pattern", nameCN: "下侧壁损伤图形", category: .infarction),
        DiseaseCondition(id: 139, nameEN: "rbbb and left posterior fascicular block", nameCN: "右束支传导阻滞伴左后分支传导阻滞", category: .conduction),
        DiseaseCondition(id: 140, nameEN: "anterolateral injury pattern", nameCN: "前侧壁损伤图形", category: .infarction),
        DiseaseCondition(id: 141, nameEN: "atrial-paced complexes", nameCN: "心房起搏波群", category: .pacemaker),
        DiseaseCondition(id: 142, nameEN: "with sinus pause", nameCN: "伴窦性停搏", category: .rhythm),
        DiseaseCondition(id: 143, nameEN: "biventricular hypertrophy", nameCN: "双心室肥大", category: .hypertrophy),
        DiseaseCondition(id: 144, nameEN: "abnormal right axis deviation", nameCN: "异常电轴右偏", category: .axis),
        DiseaseCondition(id: 145, nameEN: "supraventricular complexes", nameCN: "室上性波群", category: .rhythm),
        DiseaseCondition(id: 146, nameEN: "with 2nd degree av block mobitz i", nameCN: "伴二度I型房室传导阻滞", category: .conduction),
        DiseaseCondition(id: 147, nameEN: "with 2:1 av conduction", nameCN: "伴2:1房室传导", category: .conduction),
        DiseaseCondition(id: 148, nameEN: "with av dissociation", nameCN: "伴房室分离", category: .conduction),
        DiseaseCondition(id: 149, nameEN: "multifocal atrial tachycardia", nameCN: "多源性房性心动过速", category: .rhythm),
    ]
}
