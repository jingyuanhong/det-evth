# det-evth (Detect Everything)

**ECG Disease Screening App for iOS**

© 2026 minuscule health Ltd, London, United Kingdom

---

## Overview

**det-evth** is a consumer wellness iOS application that performs AI-powered ECG disease screening using the ECGFounder deep learning model. The app analyzes single-lead ECG recordings and screens for **150 cardiac conditions** with on-device inference.

> ⚠️ **DISCLAIMER**: This is a wellness app, NOT a medical device. Results are AI-generated and may not be accurate. Always consult a healthcare professional. Do not use for diagnosis or treatment decisions.

---

## Features

| Feature | Description |
|---------|-------------|
| **Apple Watch ECG** | Read ECG recordings directly from HealthKit |
| **PDF/Image Import** | Extract ECG waveforms from exported PDFs or photos |
| **150-Class Screening** | On-device AI analysis for cardiac conditions |
| **Bilingual Support** | Full English and Chinese (简体中文) localization |
| **PDF Reports** | Generate shareable medical reports |
| **Emergency Contacts** | Quick access to doctor and emergency contacts |
| **Heart Rate Trends** | Track heart rate statistics over time |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              det-evth iOS App                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PRESENTATION LAYER                            │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │   Home   │ │  Record  │ │  Import  │ │  History │ │ Profile  │   │    │
│  │  │   View   │ │   View   │ │   View   │ │   View   │ │   View   │   │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │    │
│  │       │            │            │            │            │          │    │
│  │       └────────────┴─────┬──────┴────────────┴────────────┘          │    │
│  └──────────────────────────┼───────────────────────────────────────────┘    │
│                             │                                                 │
│  ┌──────────────────────────▼───────────────────────────────────────────┐    │
│  │                        SERVICE LAYER                                  │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐     │    │
│  │  │                   ScreeningService                           │     │    │
│  │  │              (Orchestrates complete workflow)                │     │    │
│  │  └──────┬─────────────┬─────────────┬─────────────┬────────────┘     │    │
│  │         │             │             │             │                   │    │
│  │    ┌────▼────┐  ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐            │    │
│  │    │HealthKit│  │  Image    │ │   ECG     │ │  CoreML   │            │    │
│  │    │ Service │  │ Extractor │ │Preprocessor│ │ Inference │            │    │
│  │    └────┬────┘  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘            │    │
│  │         │             │             │             │                   │    │
│  │    ┌────▼─────────────▼─────────────▼─────────────▼────┐             │    │
│  │    │              PDFReportGenerator                    │             │    │
│  │    └───────────────────────────────────────────────────┘             │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA LAYER                                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │   │
│  │  │   Screening     │  │   Emergency     │  │   UserProfile   │        │   │
│  │  │   Repository    │  │   Contacts Repo │  │   Repository    │        │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘        │   │
│  │           └───────────────────┬┴────────────────────┘                 │   │
│  │                               │                                        │   │
│  │                    ┌──────────▼──────────┐                            │   │
│  │                    │      Core Data      │                            │   │
│  │                    │  (SQLite Database)  │                            │   │
│  │                    └─────────────────────┘                            │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                          ML LAYER                                      │   │
│  │                                                                        │   │
│  │    ┌─────────────────────────────────────────────────────────────┐    │   │
│  │    │           ECGFounder1Lead.mlmodelc (~90MB)                  │    │   │
│  │    │         150-class 1D CNN (4-bit quantized)                  │    │   │
│  │    │    Input: (1, 1, 5000) → Output: 150 probabilities          │    │   │
│  │    └─────────────────────────────────────────────────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ECG Processing Pipeline

```
                           ECG INPUT SOURCES
    ┌──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌────────┐      ┌────────────┐     ┌────────────┐     ┌────────────┐
│ Apple  │      │   PDF      │     │   Camera   │     │   Photo    │
│ Watch  │      │  Import    │     │  Capture   │     │  Library   │
│ (512Hz)│      │  (300DPI)  │     │            │     │            │
└───┬────┘      └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
    │                 │                  │                  │
    │    ┌────────────┴──────────────────┴──────────────────┘
    │    │
    │    ▼
    │  ┌─────────────────────────────────────────────────────┐
    │  │              ECG IMAGE EXTRACTOR                     │
    │  │  ┌─────────────────────────────────────────────┐    │
    │  │  │ 1. PDF Rendering (300 DPI)                  │    │
    │  │  │ 2. Red Pixel Detection (HSV + BGR)          │    │
    │  │  │ 3. Strip Detection (Horizontal Projection)  │    │
    │  │  │ 4. Waveform Extraction (Column Scanning)    │    │
    │  │  │ 5. Gap Interpolation                        │    │
    │  │  └─────────────────────────────────────────────┘    │
    │  └──────────────────────────┬──────────────────────────┘
    │                             │
    ▼                             ▼
┌───────────────────────────────────────────────────────────────┐
│                    ECG PREPROCESSOR                            │
│                                                                │
│   Raw Signal (~330-512 Hz, variable length)                   │
│         │                                                      │
│         ▼                                                      │
│   ┌─────────────────────────────────────────┐                 │
│   │ 1. RESAMPLE TO 500 Hz                   │                 │
│   │    - Linear interpolation               │                 │
│   │    - Target: 5000 samples/10s           │                 │
│   └────────────────┬────────────────────────┘                 │
│                    ▼                                           │
│   ┌─────────────────────────────────────────┐                 │
│   │ 2. BANDPASS FILTER (0.67 - 40 Hz)       │                 │
│   │    - Butterworth, Order N=4             │                 │
│   │    - Zero-phase (forward + backward)    │                 │
│   └────────────────┬────────────────────────┘                 │
│                    ▼                                           │
│   ┌─────────────────────────────────────────┐                 │
│   │ 3. NOTCH FILTER (50 Hz, Q=30)           │                 │
│   │    - Power-line interference removal    │                 │
│   │    - IIR biquad filter                  │                 │
│   └────────────────┬────────────────────────┘                 │
│                    ▼                                           │
│   ┌─────────────────────────────────────────┐                 │
│   │ 4. BASELINE REMOVAL                      │                 │
│   │    - Median filter (kernel=0.4s)        │                 │
│   │    - Subtract baseline wander           │                 │
│   └────────────────┬────────────────────────┘                 │
│                    ▼                                           │
│   ┌─────────────────────────────────────────┐                 │
│   │ 5. Z-SCORE NORMALIZATION                │                 │
│   │    - (x - μ) / σ                        │                 │
│   └────────────────┬────────────────────────┘                 │
│                    ▼                                           │
│   Preprocessed Signal: [Float] × 5000                         │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                    COREML INFERENCE                            │
│                                                                │
│   ┌─────────────────────────────────────────────────────┐     │
│   │              ECGFounder1Lead Model                   │     │
│   │                                                      │     │
│   │   Input:  MLMultiArray (1, 1, 5000) float32         │     │
│   │                      │                               │     │
│   │                      ▼                               │     │
│   │   ┌──────────────────────────────────────────┐      │     │
│   │   │  7-Stage 1D CNN with SE Blocks           │      │     │
│   │   │  Base: 64 → 64 → 160 → 160 → 400 →      │      │     │
│   │   │        400 → 1024 → 1024 filters         │      │     │
│   │   │  Kernel: 16, Stride: 2                   │      │     │
│   │   │  Activation: Swish                       │      │     │
│   │   └──────────────────────────────────────────┘      │     │
│   │                      │                               │     │
│   │                      ▼                               │     │
│   │   Output: 150 logits → sigmoid → probabilities      │     │
│   └─────────────────────────────────────────────────────┘     │
│                                                                │
│   Compute Units: CPU + Neural Engine                          │
│   Quantization: 4-bit palettization (~90MB)                   │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                    SCREENING RESULT                            │
│                                                                │
│   ┌─────────────────────────────────────────────────────┐     │
│   │  Top 10 Conditions (threshold > 10%)                │     │
│   │  ┌────────────────────────────────────────────┐    │     │
│   │  │  1. Normal Sinus Rhythm ............. 92.3% │    │     │
│   │  │  2. Sinus Arrhythmia ................ 45.2% │    │     │
│   │  │  3. Left Axis Deviation ............. 23.1% │    │     │
│   │  │  ...                                        │    │     │
│   │  └────────────────────────────────────────────┘    │     │
│   └─────────────────────────────────────────────────────┘     │
│                                                                │
│   Categories: Rhythm | Conduction | Infarction | Hypertrophy  │
│               ST/T Wave | Pacemaker | Axis | Other            │
└───────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
det-evth/
├── DetEvth/                          # iOS App
│   ├── DetEvth/
│   │   ├── App/
│   │   │   └── DetEvthApp.swift      # App entry point (@main)
│   │   │
│   │   ├── Core/
│   │   │   ├── Extensions/
│   │   │   ├── Utilities/
│   │   │   │   └── Constants.swift   # App-wide configuration
│   │   │   └── Protocols/
│   │   │
│   │   ├── Features/                 # UI Modules (SwiftUI)
│   │   │   ├── Home/
│   │   │   │   ├── HomeView.swift
│   │   │   │   └── HomeViewModel.swift
│   │   │   ├── Record/
│   │   │   │   └── RecordView.swift
│   │   │   ├── Import/
│   │   │   │   └── ImportView.swift
│   │   │   ├── Screening/
│   │   │   │   └── ScreeningService.swift
│   │   │   ├── History/
│   │   │   │   └── HistoryView.swift
│   │   │   ├── ResultDetail/
│   │   │   │   └── ResultDetailView.swift
│   │   │   ├── Profile/
│   │   │   │   └── ProfileView.swift
│   │   │   └── Settings/
│   │   │       └── SettingsView.swift
│   │   │
│   │   ├── Services/                 # Business Logic
│   │   │   ├── HealthKit/
│   │   │   │   └── HealthKitService.swift
│   │   │   ├── ECGExtraction/
│   │   │   │   └── ECGImageExtractor.swift
│   │   │   ├── CoreML/
│   │   │   │   ├── ECGInferenceService.swift
│   │   │   │   └── ECGPreprocessor.swift
│   │   │   └── Report/
│   │   │       └── PDFReportGenerator.swift
│   │   │
│   │   ├── Data/                     # Data Layer
│   │   │   ├── CoreData/
│   │   │   │   ├── PersistenceController.swift
│   │   │   │   └── CoreDataModels.swift
│   │   │   ├── Models/
│   │   │   │   ├── DiseaseCondition.swift    # 150 conditions
│   │   │   │   └── Disclaimer.swift
│   │   │   └── Repositories/
│   │   │       └── ScreeningRepository.swift
│   │   │
│   │   └── Resources/
│   │       ├── Assets.xcassets/
│   │       ├── Localizable/
│   │       │   ├── en.lproj/Localizable.strings
│   │       │   └── zh-Hans.lproj/Localizable.strings
│   │       └── ML/
│   │           └── ECGFounder1Lead_4bit.mlpackage
│   │
│   └── scripts/
│       └── convert_to_coreml.py      # Model conversion script
│
├── ocr_module/                       # Python ECG Extraction
│   ├── src/
│   │   ├── ecg_ocr.py               # Main OCR processor
│   │   ├── ecg_ocr_improved.py      # Enhanced version
│   │   └── test_ecg_ocr.py          # Test suite
│   ├── extract_and_screen.py        # End-to-end pipeline
│   ├── run_ecg_screening.py         # Screening script
│   ├── data/                        # Test ECG images
│   └── output/                      # Extracted signals
│
└── github/
    └── ECGFounder/                  # Original model code
        ├── net1d.py                 # Model architecture
        ├── util.py                  # Preprocessing functions
        └── tasks.txt                # 150 disease labels
```

---

## Core Data Schema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CORE DATA ENTITIES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐         ┌─────────────────────┐                │
│  │   ECGRecordEntity   │         │ScreeningResultEntity│                │
│  ├─────────────────────┤         ├─────────────────────┤                │
│  │ id: UUID            │────────▶│ id: UUID            │                │
│  │ createdAt: Date     │   1:1   │ createdAt: Date     │                │
│  │ source: String      │         │ primaryConditionIdx │                │
│  │ sampleRate: Float   │         │ primaryConfidence   │                │
│  │ durationSeconds     │         │ probabilitiesData   │                │
│  │ rawSignalData: Data │         │ modelVersion        │                │
│  │ originalFileName    │         └─────────────────────┘                │
│  │ notes: String?      │                                                 │
│  └─────────────────────┘                                                 │
│                                                                          │
│  ┌─────────────────────┐         ┌─────────────────────┐                │
│  │EmergencyContactEntity│         │  UserProfileEntity  │                │
│  ├─────────────────────┤         ├─────────────────────┤                │
│  │ id: UUID            │         │ id: UUID            │                │
│  │ name: String        │         │ dateOfBirth: Date?  │                │
│  │ phoneNumber: String │         │ gender: String?     │                │
│  │ relationship: String│         │ heightCM: Float     │                │
│  │ email: String?      │         │ weightKG: Float     │                │
│  │ isDoctor: Bool      │         │ doctorEmail: String?│                │
│  │ sortOrder: Int16    │         │ createdAt: Date     │                │
│  │ createdAt: Date     │         │ updatedAt: Date     │                │
│  └─────────────────────┘         └─────────────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Disease Categories (150 Classes)

| Category | Count | Examples |
|----------|-------|----------|
| **Rhythm** | 25 | Sinus Rhythm, Atrial Fibrillation, Sinus Bradycardia |
| **Conduction** | 20 | LBBB, RBBB, 1st/2nd/3rd Degree AV Block |
| **Infarction** | 18 | Anterior MI, Inferior MI, Lateral MI |
| **Hypertrophy** | 12 | LVH, RVH, Left/Right Atrial Enlargement |
| **ST/T Wave** | 25 | ST Elevation, ST Depression, T Wave Inversion |
| **Pacemaker** | 10 | Ventricular Paced, Atrial Paced, Dual Paced |
| **Axis** | 8 | Left/Right Axis Deviation, Northwest Axis |
| **Other** | 32 | Low Voltage, Prolonged QT, Early Repolarization |

All conditions include bilingual labels (English + Chinese 简体中文).

---

## Technical Specifications

### ECG Signal Processing

| Parameter | Value |
|-----------|-------|
| Target Sample Rate | 500 Hz |
| Signal Length | 5000 samples (10 seconds) |
| Bandpass Filter | 0.67 - 40 Hz (Butterworth N=4) |
| Notch Filter | 50 Hz (Q=30) |
| Baseline Kernel | 0.4 seconds (201 samples) |

### CoreML Model

| Attribute | Value |
|-----------|-------|
| Model Name | ECGFounder1Lead |
| Architecture | 1D CNN with SE Blocks |
| Input Shape | (1, 1, 5000) |
| Output | 150 sigmoid probabilities |
| Full Size | ~353 MB |
| Quantized Size | ~90 MB (4-bit palettization) |
| Compute Units | CPU + Neural Engine |

### Image Extraction

| Parameter | Value |
|-----------|-------|
| PDF Rendering | 300 DPI |
| Red Detection (BGR) | R≥150, G≤180, B≤180 |
| Red Detection (HSV) | H: 0-10° or 150-180°, S≥50, V≥150 |
| Min Strip Height | 30 pixels |
| Gap Tolerance | 30 pixels |

---

## Requirements

### iOS App

- **iOS**: 16.0+
- **Xcode**: 15.0+
- **Swift**: 5.9+
- **Device**: iPhone (with Apple Watch for HealthKit ECG)

### Python OCR Module

```bash
pip install -r ocr_module/requirements.txt
```

- Python 3.9+
- OpenCV, Pillow, scikit-image
- SciPy, NumPy, Pandas
- PyTorch (for model conversion)
- coremltools (for CoreML export)

---

## Setup Instructions

### 1. Xcode Project Setup

1. Open the existing Xcode project at `DetEvth/detevth.xcodeproj`
2. Verify all Swift files under `DetEvth/DetEvth/` are in the target
3. Ensure `ECGFounder1Lead_4bit.mlpackage` is in the target membership
4. Configure entitlements:
   - HealthKit (including Clinical Health Records)
   - App Groups (for watchOS)

### 3. Info.plist Entries

```xml
<key>NSHealthShareUsageDescription</key>
<string>Read ECG and heart rate data from Apple Watch</string>
<key>NSCameraUsageDescription</key>
<string>Capture photos of ECG printouts for analysis</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Import ECG images from your photo library</string>
```

### 3. Build and Run

```bash
# Build for device (simulator doesn't support HealthKit ECG)
xcodebuild -scheme DetEvth -destination 'platform=iOS,name=Your iPhone'
```

---

## Testing

### Python ECG Extraction

```bash
cd ocr_module
python extract_and_screen.py data/scan-apple-heart-3.jpg
```

### iOS Unit Tests

```bash
xcodebuild test -scheme DetEvth -destination 'platform=iOS Simulator,name=iPhone 15'
```

---

## License

Proprietary - © 2026 minuscule health Ltd. All rights reserved.

---

## Contact

**minuscule health Ltd**
London, United Kingdom
2026

---

## Acknowledgments

- **ECGFounder Model**: Based on research from the original ECGFounder paper
- **Apple HealthKit**: ECG data access from Apple Watch
- **CoreML**: On-device machine learning framework
