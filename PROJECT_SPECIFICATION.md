# ECG Disease Detection Mobile App - Project Specification

## Executive Summary

This project aims to develop an iOS mobile application ecosystem (iPhone + Apple Watch) that leverages Apple Watch's Lead-I ECG measurement capabilities to detect potential cardiovascular diseases using the ECGFounder foundation model. The system will process ECG signals through advanced AI algorithms to provide users with early disease detection and health monitoring.

---

## 1. Project Overview

### 1.1 Objective
Develop a mobile health application that:
- Captures Lead-I ECG signals from Apple Watch
- Processes raw ECG data and/or ECG images/PDFs
- Detects multiple cardiovascular diseases using AI foundation model
- Provides user-friendly health insights and recommendations
- Maintains medical-grade accuracy and regulatory compliance

### 1.2 Target Platforms
- **Primary**: iOS (iPhone) + watchOS (Apple Watch)
- **Minimum Requirements**:
  - iOS 16+ (for HKElectrocardiogram API access)
  - watchOS 7+ (for ECG recording)
  - Apple Watch Series 4 or later (ECG capability)

### 1.3 Key Technologies
- **ECG Foundation Model**: [ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder) (NEJM AI 2025)
- **Data Source**: Apple HealthKit HKElectrocardiogram API
- **Signal Processing**: Custom OCR + Digital Signal Processing pipeline
- **Deployment**: On-device inference (privacy-first approach)

---

## 2. Technical Architecture

### 2.1 Data Acquisition Layer

#### 2.1.1 Raw ECG Signal (Primary Path)
**Source**: Apple HealthKit `HKElectrocardiogram` API

**Capabilities**:
- Direct access to raw voltage measurements via `HKElectrocardiogramQuery`
- Native sampling frequency from Apple Watch ECG sensor
- Timestamp-aligned voltage data points
- Available since iOS 14/watchOS 7 (2020); app targets iOS 16+

**Technical Specifications** (from sample):
- **Sampling Rate**: 512 Hz (native Apple Watch)
- **Amplitude Resolution**: 10 mm/mV
- **Recording Speed**: 25 mm/s
- **Duration**: Typically 30 seconds per recording
- **Lead Type**: Single-lead (Lead-I equivalent)
- **Device**: Apple Watch Series 6/7/8/9/Ultra (watchOS 9.3.1+)

**Implementation Requirements**:
```swift
// Request HealthKit authorization
HKHealthStore.requestAuthorization(toShare: [], read: [HKElectrocardiogramType])

// Query ECG samples
HKElectrocardiogramQuery(ecgSample) { query, result in
    // Process voltage measurements
    result.voltageMeasurements // Array of (timestamp, voltage) tuples
}
```

#### 2.1.2 ECG Image/PDF (Secondary Path - OCR)
**Source**: Apple Health app exports, Apple Watch PDF reports

**Use Cases**:
- Historical ECG data already stored as PDF
- Batch processing of existing records
- Backup method when raw signal unavailable
- Support for legacy Apple Watch recordings

**OCR Requirements**:
- **Input Format**: PDF or image (PNG, JPEG) with grid-based ECG waveform
- **Grid Detection**: Identify standard ECG grid (5mm x 5mm squares)
- **Signal Extraction**: Convert red waveform pixels to digital signal
- **Calibration**: Use grid and metadata (25mm/s, 10mm/mV) for accurate scaling
- **Multi-strip Processing**: Handle 3-strip ECG layouts (0-10s, 10-20s, 20-30s)

**Sample ECG Metadata** (extracted from PDF):
```
Patient Info: Name, DOB, Age
Recording Time: 2023-04-02 15:26
Heart Rate: Average 109 bpm
Technical Specs: 25 mm/s, 10 mm/mV, Lead-I, 512Hz
Device: iOS 26.2, watchOS 9.3.1, Watch 6,14
Classification: High heart rate (房颤显示)
```

---

### 2.2 Signal Processing Pipeline

#### 2.2.1 Preprocessing Requirements (ECGFounder Model)
Based on the foundation model specifications:

**Step 1: Resampling**
- **Target Frequency**: 500 Hz (from 512 Hz native)
- **Method**: Linear interpolation
- **Purpose**: Standardize sampling rate across different ECG sources

**Step 2: Filtering**
- **High-Pass Filter**: 0.5 Hz cutoff (suppress baseline drift)
- **Low-Pass Filter**: 50 Hz Butterworth (2nd order) - reduce high-frequency noise
- **Notch Filter**: 50/60 Hz - eliminate electrical interference

**Step 3: Segmentation**
- **Window Size**: 10 seconds
- **Strategy**:
  - For signals > 10s: Extract sequential 10-second windows
  - For signals < 10s: Apply zero-padding to reach 10 seconds
- **Sample Count**: 5000 samples per segment (500 Hz × 10s)

**Step 4: Normalization**
- **Method**: Z-score normalization (per segment)
- **Formula**: `(signal - mean(signal)) / (std(signal) + 1e-8)`
- **Purpose**: Remove inter-subject variability in amplitude

**Implementation Reference**: `dataset.py` from ECGFounder repository

#### 2.2.2 OCR Signal Extraction Pipeline

**Phase 1: Image Preprocessing**
1. PDF to image conversion (if needed)
2. Grid detection and perspective correction
3. Red waveform isolation (color-based segmentation)
4. Noise reduction and binarization

**Phase 2: Waveform Digitization**
1. Horizontal strip detection (identify 3 strips)
2. Baseline identification for each strip
3. Pixel-to-voltage conversion using calibration metadata
4. Time-series reconstruction (25mm/s → sample timestamps)
5. Multi-strip concatenation (0-30s continuous signal)

**Phase 3: Quality Control**
1. Detect missing segments or artifacts
2. Validate signal continuity across strips
3. Check amplitude ranges (physiological plausibility)
4. Compare with reported heart rate (metadata validation)

**Output**: Continuous 30-second ECG signal at 512 Hz → Ready for preprocessing pipeline

---

### 2.3 AI/ML Model Layer

#### 2.3.1 ECGFounder Foundation Model

**Model Details**:
- **Architecture**: Transformer-based foundation model
- **Training Data**: 10,771,552 ECGs from 1,818,247 subjects
- **Dataset**: Harvard-Emory ECG Database
- **Label Categories**: 150 cardiovascular conditions
- **Publication**: NEJM AI (January 2025)
- **Repository**: [PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder)
- **Model Weights**: [HuggingFace](https://huggingface.co/PKUDigitalHealth/ECGFounder)

**Performance Metrics**:
- **AUROC > 0.95** for 80 diagnoses (expert-level performance)
- **Internal validation**: High accuracy on Harvard-Emory test set
- **External validation**: Proven generalization across multiple domains
- **Fine-tuning gains**: +3-5% AUROC improvement on downstream tasks

**Supported Lead Configurations**:
- Full 12-lead ECG (standard clinical setup)
- **Single-lead ECG** (Lead-I) - **Perfect for Apple Watch!**
- Reduced-lead configurations for mobile/remote monitoring

#### 2.3.2 Disease Detection Capabilities

**Primary Detection Tasks** (from research paper):

1. **Cardiovascular Diagnoses** (150 categories, 80 with AUROC > 0.95)
   - Atrial fibrillation (AFib)
   - Atrial flutter
   - Various arrhythmias
   - Conduction abnormalities
   - Ischemic changes
   - Hypertrophy patterns
   - [Full list: 150 diagnostic categories]

2. **Demographics Detection**
   - Age prediction (regression)
   - Sex classification

3. **Clinical Event Detection**
   - **Chronic Kidney Disease (CKD)** detection
   - **Chronic Heart Disease (CHD)** detection
   - **Left Ventricular Ejection Fraction (LVEF)** regression & classification
   - **NT-proBNP levels** regression & abnormality detection

4. **Cross-Modality Applications**
   - Atrial fibrillation detection from PPG (photoplethysmography)
   - Enables smartwatch-based continuous monitoring

**Note**: Full disease list available in `tasks.txt` in ECGFounder repository (to be verified during implementation).

#### 2.3.3 On-Device Deployment Strategy

**Challenges**:
- Large foundation model size (millions of parameters)
- Real-time inference requirements
- Limited mobile device computational resources
- Battery consumption constraints

**Proposed Solutions**:

1. **Model Compression**
   - Quantization (INT8/FP16) to reduce model size
   - Pruning redundant weights
   - Knowledge distillation to smaller student model

2. **CoreML Integration**
   - Convert PyTorch model to CoreML format
   - Leverage Apple Neural Engine for acceleration
   - Optimize for iOS hardware (A-series chips)

3. **Hybrid Approach** (if on-device is infeasible)
   - On-device: Lightweight screening model (high sensitivity)
   - Cloud-based: Full ECGFounder model for detailed analysis
   - Privacy-preserving: Encrypted signal transmission

4. **Inference Optimization**
   - Batch processing of historical ECGs
   - Background processing during charging
   - Progressive loading (prioritize recent recordings)

---

## 3. System Components & Features

### 3.1 Apple Watch Component

**Responsibilities**:
- ECG signal acquisition (user-initiated)
- Real-time quality feedback during recording
- Basic heart rate monitoring
- Data synchronization to iPhone via HealthKit

**User Flow**:
1. User opens ECG app on Apple Watch
2. Places finger on Digital Crown
3. Records 30-second ECG
4. Data automatically synced to iPhone via HealthKit
5. Notification when analysis completes

### 3.2 iPhone Application

**Core Features**:

#### 3.2.1 Data Management
- **ECG Import**:
  - Automatic HealthKit sync (raw signals)
  - Manual PDF/image upload (OCR path)
  - Historical data retrieval
- **Storage**: Local encrypted database (HIPAA/GDPR compliance)
- **Export**: PDF reports, CSV data, FHIR-compatible formats

#### 3.2.2 Signal Processing
- Background processing queue for new ECG recordings
- OCR engine for PDF/image conversion
- Preprocessing pipeline (filtering, normalization)
- Quality assessment and artifact detection

#### 3.2.3 AI Inference
- On-device model inference (CoreML)
- Multi-disease probability scores
- Confidence intervals and uncertainty quantification
- Trend analysis (compare with historical recordings)

#### 3.2.4 User Interface
- **Dashboard**:
  - Recent ECG recordings timeline
  - Health status summary
  - Risk alerts and recommendations
- **ECG Viewer**:
  - Interactive waveform visualization
  - Zoom/pan controls
  - Annotation tools (P-QRS-T wave markers)
- **Analysis Results**:
  - Disease detection probabilities
  - Risk stratification (low/medium/high)
  - Educational content (what is AFib, etc.)
  - Actionable recommendations
- **History & Trends**:
  - Long-term heart health tracking
  - Statistical summaries
  - Export and sharing options

#### 3.2.5 Clinical Decision Support
- **Risk Alerts**:
  - Critical findings flagged immediately
  - Push notifications for high-risk conditions
- **Recommendations**:
  - "Consult cardiologist" for concerning patterns
  - "Repeat test in X weeks" for monitoring
  - "No abnormalities detected" for normal results
- **Disclaimers**:
  - Clear medical device limitations
  - Not a replacement for professional diagnosis
  - Emergency contact information (when to call 911)

### 3.3 Backend Services (Optional Cloud Component)

**If hybrid approach adopted**:
- Secure API for model inference
- User authentication and authorization
- Anonymized data for model improvement (opt-in)
- Telemetry and crash reporting

---

## 4. Development Phases

### Phase 1: Foundation & Research (Current)
- [x] Project specification document
- [ ] OCR algorithm development and validation
  - [ ] Grid detection module
  - [ ] Waveform extraction from ECG images
  - [ ] Calibration and scaling algorithms
  - [ ] Multi-strip concatenation logic
  - [ ] Accuracy validation against ground truth signals
- [ ] HealthKit integration prototype
  - [ ] HKElectrocardiogram API authorization
  - [ ] Raw voltage data extraction
  - [ ] Data format conversion and storage
- [ ] Signal preprocessing pipeline implementation
  - [ ] Resampling (512 Hz → 500 Hz)
  - [ ] Filter design (high-pass, low-pass, notch)
  - [ ] Segmentation (10-second windows)
  - [ ] Z-score normalization
  - [ ] Validation against ECGFounder requirements

### Phase 2: Model Integration
- [ ] ECGFounder model setup
  - [ ] Download model weights from HuggingFace
  - [ ] Test inference with sample ECG data
  - [ ] Validate output format and disease labels
- [ ] CoreML conversion pipeline
  - [ ] PyTorch → ONNX → CoreML conversion
  - [ ] Model quantization and optimization
  - [ ] On-device inference testing
  - [ ] Performance benchmarking (latency, memory)
- [ ] Disease classification integration
  - [ ] Parse model outputs (150 disease probabilities)
  - [ ] Confidence thresholding logic
  - [ ] Multi-label classification handling

### Phase 3: iOS/watchOS App Development
- [ ] Project setup (Xcode, SwiftUI)
- [ ] Apple Watch ECG recording interface
- [ ] iPhone data synchronization
- [ ] OCR module integration (image → signal)
- [ ] Real-time signal visualization
- [ ] Model inference pipeline
- [ ] Results display and interpretation UI
- [ ] Local data storage (Core Data / SQLite)
- [ ] HealthKit permissions and privacy handling

### Phase 4: Clinical Validation & Testing
- [ ] Algorithm accuracy validation
  - [ ] Compare OCR output vs. ground truth signals
  - [ ] Measure signal reconstruction error
- [ ] Model performance evaluation
  - [ ] Test on diverse ECG recordings
  - [ ] Sensitivity/specificity analysis per disease
  - [ ] False positive/negative rate assessment
- [ ] User experience testing
  - [ ] Usability studies with target users
  - [ ] Interface refinement based on feedback
- [ ] Clinical expert review
  - [ ] Cardiologist evaluation of detected findings
  - [ ] Medical accuracy validation
  - [ ] Risk stratification appropriateness

### Phase 5: Regulatory & Deployment
- [ ] Medical device compliance (FDA/CE Mark)
  - [ ] Determine classification (Class I/II/III)
  - [ ] Prepare regulatory submissions
  - [ ] Clinical trial requirements (if applicable)
- [ ] Privacy & security compliance
  - [ ] HIPAA compliance (US)
  - [ ] GDPR compliance (EU)
  - [ ] Data encryption and secure storage
  - [ ] Privacy policy and terms of service
- [ ] App Store deployment
  - [ ] Apple App Store guidelines compliance
  - [ ] Beta testing (TestFlight)
  - [ ] Production release

---

## 5. Technical Challenges & Solutions

### 5.1 OCR Accuracy Challenge

**Problem**: Converting ECG images to digital signals with high fidelity
- Grid detection errors in poor-quality scans
- Waveform extraction noise from image artifacts
- Baseline wander in printed ECGs
- Multi-strip alignment issues

**Solutions**:
1. **Robust Grid Detection**:
   - Hough transform for line detection
   - RANSAC for perspective correction
   - Validate grid spacing (5mm × 5mm standard)
2. **Advanced Waveform Extraction**:
   - Color-space segmentation (isolate red waveform)
   - Morphological operations for noise reduction
   - Spline interpolation for smooth reconstruction
3. **Quality Metrics**:
   - Signal-to-noise ratio (SNR) estimation
   - Correlation with metadata (heart rate consistency)
   - Flag low-quality scans for manual review

### 5.2 Model Size & On-Device Inference

**Problem**: ECGFounder is a large transformer model
- Likely 50-200+ MB model size
- Inference latency on mobile devices
- Battery drain concerns

**Solutions**:
1. **Model Optimization**:
   - INT8 quantization (4x size reduction, minimal accuracy loss)
   - Distillation to smaller architecture (10-20 MB target)
   - Layer pruning for mobile deployment
2. **Apple Silicon Acceleration**:
   - CoreML Neural Engine utilization
   - Metal Performance Shaders (MPS) for GPU acceleration
3. **User Experience**:
   - Background processing (avoid blocking UI)
   - Progress indicators during inference
   - Offline-first architecture (no internet required)

### 5.3 Single-Lead Limitations

**Problem**: Apple Watch provides only Lead-I ECG
- Limited diagnostic information vs. 12-lead ECG
- Some conditions require multiple leads for diagnosis

**Solutions**:
1. **Model Capabilities**:
   - ECGFounder explicitly supports single-lead inference
   - Trained on Lead-I data for mobile scenarios
2. **Diagnostic Scope**:
   - Focus on diseases detectable from Lead-I (AFib, arrhythmias)
   - Clear disclaimers about limitations
   - Recommend 12-lead ECG for comprehensive evaluation
3. **Future Enhancement**:
   - Multi-lead synthesis (AI-generated leads from Lead-I)
   - Integration with external 12-lead ECG devices

### 5.4 Regulatory Compliance

**Problem**: Health apps with diagnostic claims require FDA approval
- Class II medical device designation likely
- Extensive clinical validation required
- High regulatory burden and cost

**Solutions**:
1. **Wellness vs. Diagnostic Positioning**:
   - Launch as "informational" tool (not diagnostic)
   - Provide "risk indicators" rather than "diagnoses"
   - Clear disclaimers: "Not a substitute for medical advice"
2. **Phased Regulatory Approach**:
   - V1: General wellness app (no FDA approval needed)
   - V2: Pursue FDA 510(k) clearance for specific indications
   - V3: Expand approved diagnostic capabilities
3. **Clinical Validation**:
   - Partner with academic medical centers
   - Publish peer-reviewed validation studies
   - Build clinical credibility for regulatory submissions

### 5.5 Privacy & Data Security

**Problem**: ECG data is highly sensitive PHI (Protected Health Information)

**Solutions**:
1. **Data Minimization**:
   - Store only necessary data locally
   - No cloud backup by default (user opt-in)
2. **Encryption**:
   - At-rest: iOS Data Protection APIs
   - In-transit: TLS 1.3+ for any network communication
3. **User Control**:
   - Granular permissions for HealthKit access
   - Export and delete functionality
   - Transparent privacy policy

---

## 6. Success Metrics

### 6.1 Technical Performance
- **OCR Accuracy**: > 95% signal reconstruction accuracy vs. raw signals
- **Inference Latency**: < 3 seconds per 30-second ECG
- **Model AUROC**: > 0.90 for top 10 target diseases (maintain ECGFounder performance)
- **App Responsiveness**: 60 FPS UI, < 200ms interaction latency

### 6.2 Clinical Validation
- **Sensitivity**: > 90% for critical conditions (AFib, MI patterns)
- **Specificity**: > 85% (minimize false alarms)
- **Agreement with Expert Cardiologists**: Cohen's kappa > 0.80

### 6.3 User Adoption
- **User Retention**: > 50% monthly active users at 6 months
- **Recording Frequency**: Average 2+ ECGs per user per month
- **User Satisfaction**: > 4.0/5.0 App Store rating

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regulatory rejection (FDA) | Medium | High | Start as wellness app, pursue clearance incrementally |
| On-device model too slow | Medium | Medium | Optimize with quantization, consider cloud fallback |
| OCR accuracy insufficient | Low | High | Prioritize raw signal path, OCR as secondary |
| Low user adoption | Medium | High | Focus on UX, clinical validation, marketing |
| Privacy breach | Low | Critical | Security audit, penetration testing, encryption |
| Medical liability | Medium | Critical | Strong disclaimers, liability insurance, legal review |
| Apple Watch API limitations | Low | Medium | Test thoroughly on multiple watchOS versions |

---

## 8. Future Enhancements (Post-V1)

### 8.1 Extended Device Support
- Android Wear integration (Samsung Galaxy Watch ECG)
- External ECG device compatibility (AliveCor Kardia, etc.)
- 12-lead ECG import from hospital systems (FHIR integration)

### 8.2 Advanced Features
- **Continuous Monitoring**: Background heart rate + irregular rhythm notifications
- **Predictive Analytics**: Risk prediction models (e.g., "20% risk of AFib in next year")
- **Medication Tracking**: Correlate ECG changes with drug regimens
- **Telemedicine Integration**: Share ECG reports with physicians via secure portal

### 8.3 Research Contributions
- **Federated Learning**: Improve model with user data (privacy-preserving)
- **Rare Disease Detection**: Expand to uncommon cardiac conditions
- **Personalized Baselines**: Learn individual "normal" patterns over time

---

## 9. References & Resources

### 9.1 Academic Papers
- **ECGFounder Paper**: [NEJM AI - An Electrocardiogram Foundation Model Built on over 10 Million Recordings](https://ai.nejm.org/doi/abs/10.1056/AIoa2401033)
- PubMed: https://pubmed.ncbi.nlm.nih.gov/40771651/
- PMC Full Text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12327759/

### 9.2 Model Resources
- **GitHub Repository**: https://github.com/PKUDigitalHealth/ECGFounder
- **HuggingFace Model**: https://huggingface.co/PKUDigitalHealth/ECGFounder
- **Key Files**:
  - `dataset.py` - Signal preprocessing pipeline
  - `tasks.txt` - Full list of 150 disease categories (to be confirmed)

### 9.3 Apple Developer Documentation
- **HKElectrocardiogram API**: https://developer.apple.com/documentation/healthkit/hkelectrocardiogram
- **HealthKit Overview**: https://developer.apple.com/documentation/healthkit
- **WWDC 2020 - What's New in HealthKit**: https://developer.apple.com/videos/play/wwdc2020/10182/
- **Reading Data from HealthKit**: https://developer.apple.com/documentation/healthkit/reading-data-from-healthkit

### 9.4 Regulatory Resources
- **FDA Digital Health Center**: https://www.fda.gov/medical-devices/digital-health-center-excellence
- **Clinical Decision Support Software Guidance**: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- **EU MDR Compliance**: https://ec.europa.eu/health/medical-devices-sector/new-regulations_en

---

## 10. Team & Expertise Required

### 10.1 Technical Roles
- **iOS/watchOS Developer** (SwiftUI, HealthKit, CoreML)
- **ML Engineer** (PyTorch, model optimization, CoreML conversion)
- **Computer Vision Engineer** (OCR algorithm development)
- **Signal Processing Engineer** (ECG filtering, artifact detection)
- **Backend Engineer** (if cloud component needed)

### 10.2 Clinical/Regulatory
- **Clinical Cardiologist** (advisory role, validation)
- **Regulatory Affairs Specialist** (FDA/CE Mark compliance)
- **Clinical Data Scientist** (validation study design)

### 10.3 Design & Product
- **UX/UI Designer** (medical app experience)
- **Product Manager** (roadmap, feature prioritization)
- **QA Engineer** (medical software testing)

---

## 11. Project Timeline Estimate

**Note**: Timeline provided for planning purposes only; actual duration depends on team size, resources, and regulatory pathway.

- **Phase 1** (Foundation): 2-3 months
- **Phase 2** (Model Integration): 1-2 months
- **Phase 3** (App Development): 3-4 months
- **Phase 4** (Validation): 2-3 months
- **Phase 5** (Regulatory & Launch): 3-6 months (highly variable)

**Total**: 11-18 months to first production release (wellness app)
**With FDA Clearance**: +6-12 months additional

---

## 12. Next Steps

### Immediate Actions (Phase 1 - Current Sprint)

1. **OCR Algorithm Development** ✓ (First Priority)
   - Implement grid detection from ECG image sample
   - Extract waveform from red pixels
   - Convert to time-series signal
   - Validate against known ECG parameters (109 bpm, 30s duration)

2. **HealthKit API Prototype**
   - Create minimal iOS app
   - Request HKElectrocardiogram permissions
   - Query and display raw voltage data
   - Compare raw vs. PDF-exported data

3. **Signal Preprocessing Pipeline**
   - Implement filtering (high-pass, low-pass, notch)
   - Resampling to 500 Hz
   - Segmentation and normalization
   - Unit testing with synthetic signals

4. **ECGFounder Model Testing**
   - Clone GitHub repository
   - Download model weights
   - Run inference on sample Lead-I ECG
   - Document input/output formats

### Decision Points

- **On-device feasibility**: By end of Phase 2, decide if on-device inference is viable or if cloud-based approach is needed
- **Regulatory strategy**: By end of Phase 3, determine if pursuing FDA clearance or launching as wellness app
- **OCR priority**: If HealthKit raw signal path proves robust, OCR may be deprioritized to Phase 2

---

## Document Control

- **Version**: 1.0
- **Date**: 2026-01-16
- **Author**: Project Team
- **Status**: Draft - Awaiting Stakeholder Review
- **Next Review**: After Phase 1 OCR prototype completion

---

## Appendix A: ECG Image Sample Analysis

**File**: Provided PDF (洪静远 ECG Report)

**Key Observations**:
- **Format**: 3-strip layout, standard ECG grid background
- **Recording**: 30 seconds total (0-10s, 10-20s, 20-30s strips)
- **Waveform Color**: Red line on pink/gray grid
- **Calibration**: Scale bar visible (top-left)
- **Metadata**: Chinese + English text, patient info, technical parameters
- **Heart Rate**: 109 bpm (labeled as "高心率" - high heart rate)
- **Quality**: High-quality digital export from Apple Watch app

**OCR Challenges**:
- Multi-strip requires careful alignment
- Baseline detection across strips (slight vertical offset)
- Grid lines close to waveform (filtering needed)
- Chinese characters in metadata (internationalization consideration)

**OCR Opportunities**:
- Clear red waveform (good color contrast)
- Standard ECG grid (well-defined 5mm spacing)
- High resolution (sufficient for accurate digitization)
- Metadata provides ground truth for validation (109 bpm, 512 Hz)

---

## Appendix B: Glossary

- **Lead-I ECG**: Bipolar limb lead measuring electrical potential difference between right arm and left arm
- **AUROC**: Area Under Receiver Operating Characteristic curve (classification performance metric)
- **HKElectrocardiogram**: Apple HealthKit class for ECG sample data
- **CoreML**: Apple's machine learning framework for on-device inference
- **OCR**: Optical Character Recognition (here: image-to-signal conversion)
- **SNR**: Signal-to-Noise Ratio
- **PHI**: Protected Health Information (HIPAA term)
- **FDA 510(k)**: Premarket notification for medical devices
- **FHIR**: Fast Healthcare Interoperability Resources (health data standard)

---

**End of Document**
