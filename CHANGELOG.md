# det-evth changelog

## 2026-01-23

### session 5: company logo integration

#### overview
- added company logo (logo_clean.png - ecg waveform only) to the app's settings/about section
- company name "minuscule health ltd." displayed as text below the logo
- updated all 3 project variants (DetEvth, DetEvth-iOS, DetEvth-App)

#### asset catalog updates
created CompanyLogo.imageset in each project:
- `DetEvth/detevth/detevth/Assets.xcassets/CompanyLogo.imageset/`
- `DetEvth-iOS/Assets.xcassets/CompanyLogo.imageset/`
- `DetEvth-App/Assets.xcassets/CompanyLogo.imageset/`

#### SettingsView.swift updates
- added company logo (ecg waveform) at 40pt height
- added "minuscule health ltd." text below logo in caption/secondary style
- added "© 2026" text below company name in caption2/tertiary style
- centered layout in about section with 4pt spacing

files updated:
- `DetEvth/detevth/detevth/Features/Settings/SettingsView.swift`
- `DetEvth-iOS/Features/Settings/SettingsView.swift`
- `DetEvth-App/Features/Settings/SettingsView.swift`

#### exclude "abnormal ecg" from results display
- modified result processing to skip index 0 ("abnormal ecg") in all locations:
  - `ScreeningResult.primaryCondition` - inference service
  - `ScreeningResult.topConditions()` - inference service
  - `ScreeningRepository.saveScreeningResult()` - when saving to Core Data
  - `ScreeningResultEntity.create()` - Core Data convenience initializer
  - `ScreeningResultEntity.topConditions()` - Core Data entity method
- when model outputs "abnormal ecg" as highest probability, the second-highest condition is shown instead
- this is a display-only change; model output and raw probabilities are unchanged

files updated:
- `DetEvth/detevth/detevth/Services/CoreML/ECGInferenceService.swift`
- `DetEvth/detevth/detevth/Data/Repositories/ScreeningRepository.swift`
- `DetEvth/detevth/detevth/Data/CoreData/CoreDataModels.swift`
- `DetEvth/detevth/detevth/Features/ResultDetail/ResultDetailView.swift` (TopConditionsSection)
- `DetEvth-iOS/Services/CoreML/ECGInferenceService.swift`
- `DetEvth-iOS/Data/Repositories/ScreeningRepository.swift`
- `DetEvth-iOS/Data/CoreData/CoreDataModels.swift`
- `DetEvth-App/Services/CoreML/ECGInferenceService.swift`
- `DetEvth-App/Data/Repositories/ScreeningRepository.swift`
- `DetEvth-App/Data/CoreData/CoreDataModels.swift`

#### disease label localization fix (CN mode)
- changed `.nameEN` to `.localizedName` for all disease condition displays
- disease labels now show Chinese names when device is in CN mode
- added `common.unknown` localized string (en: "unknown", cn: "未知")
- updated `isNormalCondition` check to support both EN and CN condition names

files updated:
- `DetEvth/detevth/detevth/Features/ResultDetail/ResultDetailView.swift`
- `DetEvth/detevth/detevth/Features/History/HistoryView.swift`
- `DetEvth/detevth/detevth/Features/Import/ImportView.swift`
- `DetEvth/detevth/detevth/Features/Home/HomeViewModel.swift`
- `DetEvth/detevth/detevth/Resources/Localizable/en.lproj/Localizable.strings`
- `DetEvth/detevth/detevth/Resources/Localizable/zh-Hans.lproj/Localizable.strings`

#### primary finding icon logic update
- changed icon/color logic in `PrimaryFindingCard` from confidence-based to condition-based
- green checkmark: shown only for "sinus rhythm" or "normal sinus rhythm" (normal conditions)
- orange exclamation: shown for all other conditions (abnormal findings)
- previous behavior was based on confidence > 0.8

files updated:
- `DetEvth/detevth/detevth/Features/ResultDetail/ResultDetailView.swift`

#### edit/done button updates
- removed edit/done button from HistoryView (swipe-to-delete is sufficient)
- replaced `.onDelete` with custom `.swipeActions` using `common.delete` ("delete") for lowercase
- ProfileView: replaced system `EditButton()` with custom button using `common.edit`/`common.done`

files updated:
- `DetEvth/detevth/detevth/Features/History/HistoryView.swift` (removed edit button, lowercase delete)
- `DetEvth/detevth/detevth/Features/Profile/ProfileView.swift`
- `DetEvth-iOS/Features/Profile/ProfileView.swift`
- `DetEvth-App/Features/Profile/ProfileView.swift`

#### profile section headers lowercase
- added `.textCase(nil)` to section headers to prevent SwiftUI auto-uppercasing
- headers now display as: "personal information", "emergency contacts", "my doctor's email"

files updated:
- `DetEvth/detevth/detevth/Features/Profile/ProfileView.swift`

#### app icon update
- added new app icon (app_logo.png) - navy blue background with white ECG waveform and red indicator
- updated all 3 iOS project variants (DetEvth, DetEvth-iOS, DetEvth-App)
- same icon used for regular, dark mode, and tinted variants

files updated:
- `DetEvth/detevth/detevth/Assets.xcassets/AppIcon.appiconset/`
- `DetEvth-iOS/Assets.xcassets/AppIcon.appiconset/`
- `DetEvth-App/Assets.xcassets/AppIcon.appiconset/`

---

### session 4: lowercase branding update

#### overview
- converted all english ui text to lowercase to align with minuscule health ltd branding
- updated all 3 project variants (DetEvth, DetEvth-iOS, DetEvth-App)

#### localizable.strings updates (en.lproj)
all english strings converted to lowercase including:
- tab bar: home, history, profile
- screen titles: record ecg, import ecg, ecg history, my profile, settings
- actions: record ecg, import pdf, generate report, share
- labels: heart rate today, recent screenings, personal information
- messages: all error messages, progress indicators, disclaimers
- categories: rhythm, conduction, infarction, hypertrophy, st/t wave, pacemaker, axis, other
- technical terms: ecg, pdf, ai, healthkit, apple watch (all lowercase per branding)

files updated:
- `DetEvth/detevth/detevth/Resources/Localizable/en.lproj/Localizable.strings`
- `DetEvth-iOS/Resources/Localizable/en.lproj/Localizable.strings`
- `DetEvth-App/Resources/Localizable/en.lproj/Localizable.strings`

#### DiseaseCondition.swift updates
- all 150 english disease condition names converted to lowercase
- category enum raw values converted to lowercase
- chinese translations unchanged

files updated:
- `DetEvth/detevth/detevth/Data/Models/DiseaseCondition.swift`
- `DetEvth-iOS/Data/Models/DiseaseCondition.swift`
- `DetEvth-App/Data/Models/DiseaseCondition.swift`

#### examples of changes
| before | after |
|--------|-------|
| Home | home |
| Record ECG | record ecg |
| DISCLAIMER | disclaimer |
| SINUS RHYTHM | sinus rhythm |
| Apple Watch | apple watch |
| © 2026 minuscule health Ltd | © 2026 minuscule health ltd |

---

## 2026-01-19

### session 1: initial fixes

#### overview
- updated documentation to reflect iOS 16+, existing Xcode project usage, Clinical Health Records entitlement, and prebuilt CoreML model
- fixed multiple Swift compile warnings/errors and a runtime crash due to localized string formatting
- relaunched the iOS Simulator on request

#### documentation changes
- updated setup steps and entitlements in `det-evth/README.md`
- updated minimum requirements and notes in `det-evth/PROJECT_SPECIFICATION.md`
- revised `det-evth/remote-guide.docx` to open the existing project and validate target membership (no new project creation; CoreML already provided)

#### code changes
- added `UIKit` import to resolve `UIImage` in `det-evth/DetEvth/detevth/detevth/Features/Screening/ScreeningService.swift`
- fixed HealthKit compile errors and Swift 6 warnings in `det-evth/DetEvth/detevth/detevth/Services/HealthKit/HealthKitService.swift`:
  - use `durationInSeconds` instead of missing `duration`
  - avoid tuple type inference to `[Any]` in heart rate mapping
  - avoid shadowing `max` and use `Swift.max`
  - updated observer Task to `@MainActor` and safe `self` capture
  - added `.unrecognized` case to ECG classification switch
- suppressed CoreML Sendable warnings with `@preconcurrency import CoreML` in `det-evth/DetEvth/detevth/detevth/Services/CoreML/ECGInferenceService.swift`
- cleaned PDF report warnings in `det-evth/DetEvth/detevth/detevth/Services/Report/PDFReportGenerator.swift`
- fixed localized formatting crash by using `String(format:)` and correct format specifiers:
  - `det-evth/DetEvth/detevth/detevth/Features/Home/HomeView.swift`
  - `det-evth/DetEvth/detevth/detevth/Features/History/HistoryView.swift`
  - `det-evth/DetEvth/detevth/detevth/Features/ResultDetail/ResultDetailView.swift`
  - `det-evth/DetEvth/detevth/detevth/Resources/Localizable/en.lproj/Localizable.strings`
  - `det-evth/DetEvth/detevth/detevth/Resources/Localizable/zh-Hans.lproj/Localizable.strings`

#### simulator
- restarted the iOS Simulator (resolved the `eligibility.plist` missing log)

---

### session 2: feature implementation

#### import feature fixes
- **ImportView.swift**: implemented full PDF and image import functionality
  - added `DocumentPicker` using `UIDocumentPickerViewController` for PDF file selection
  - added `onChange` handler for `PhotosPicker` to process selected images
  - removed "Take Photo" option (camera) - no image noise removal algorithm available
  - added processing overlay, error alerts, and navigation to results
  - connected to `ECGImageExtractor`, `ECGInferenceService`, and `ScreeningRepository`

#### history delete functionality
- **HistoryView.swift**: connected to Core Data for real data and deletion
  - `HistoryViewModel` now fetches from `ScreeningRepository`
  - swipe-to-delete properly removes records from Core Data
  - added Edit button in toolbar and pull-to-refresh support

#### profile editing
- **ProfileView.swift**: made all profile fields editable
  - personal info: DOB (date picker), gender (picker), height, weight
  - emergency contacts: add, edit (tap name), delete (swipe)
  - doctor's email: editable with email keyboard
  - connected to `UserProfileRepository` and `EmergencyContactsRepository`

#### ECG extraction algorithm - multi-color support
- **ECGImageExtractor.swift**: auto-detects waveform color
  - supports: red, black, blue, green waveforms
  - samples image to detect dominant color
  - improved thresholds: `minStripHeight` 30→20, `gapTolerance` 30→50
  - added color detection functions: `isRed()`, `isBlack()`, `isBlue()`, `isGreen()`

#### ResultDetailView - real data display
- **ResultDetailView.swift**: shows actual extracted data
  - waveform section displays real ECG signal (downsampled to 500 points)
  - top conditions section shows actual AI predictions (≥5% probability)
  - uses `DiseaseConditions.all` for condition name lookup

#### heart rate integration
- **HomeViewModel.swift**: connected to HealthKit and ECG-based extraction
  - fetches today's heart rate stats from HealthKit (avg, min, max)
  - shows `--` when HealthKit unavailable or no data
  - added `ECGHeartRateExtractor` for R-peak detection and BPM calculation
  - `ScreeningResultModel` now includes `heartRate: Int?` field

- **HistoryView.swift**: extracts heart rate from stored ECG signals
- **ImportView.swift**: extracts heart rate after PDF/image import

#### model updates
- **ScreeningResultModel**: added fields
  - `signal: [Float]?` - ECG waveform data
  - `probabilities: [Float]?` - all 150 disease probabilities
  - `heartRate: Int?` - extracted from ECG waveform

#### localization updates
- **en.lproj/Localizable.strings** and **zh-Hans.lproj/Localizable.strings**:
  - import processing messages
  - import error messages
  - profile editing labels
  - results waveform/conditions no-data messages

#### Info.plist configuration
- **Info.plist**: added HealthKit usage descriptions (required for HealthKit authorization)
  - `NSHealthShareUsageDescription` - for reading ECG and heart rate data
  - `NSHealthUpdateUsageDescription` - for saving screening results
  - note: add these keys in Xcode Target → Info tab to avoid duplicate Info.plist conflicts

---

### session 3: ux improvements

#### heart rate fallback from ECG
- **HomeViewModel.swift**: added fallback when HealthKit data unavailable
  - if HealthKit returns no heart rate data (or authorization fails), falls back to most recent ECG
  - uses `ECGHeartRateExtractor` to extract heart rate from stored ECG signal
  - added `loadHeartRateFromRecentECG()` method for fallback logic
  - min/max values display as `nil` when using ECG fallback (single measurement)

#### ECG waveform display improvements
- **ResultDetailView.swift**: enhanced waveform readability
  - added horizontal `ScrollView` for panning through the entire waveform
  - increased line thickness from 1 to 2.5 with rounded caps/joins
  - added `yAxisBounds` computed property for tight Y-axis range (10% padding around signal)
  - uses `.chartYScale(domain:)` to constrain Y-axis for better visibility
  - increased display height from 150 to 200 points
  - increased detail from 500 to 2000 display points for better resolution
  - chart width scales with data points (`displaySignal.count * 1.5`) for scrollable view
