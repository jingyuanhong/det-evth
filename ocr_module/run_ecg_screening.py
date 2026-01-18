#!/usr/bin/env python3
"""
ECG Disease Screening using ECGFounder Model

This script:
1. Loads extracted ECG strips from .npy files
2. Resamples from extraction rate (~330 Hz) to model rate (500 Hz)
3. Applies ECGFounder preprocessing (bandpass, notch, baseline removal, z-score)
4. Runs the 1-lead ECGFounder model for 150-class disease screening
"""

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.signal import resample, medfilt, iirnotch, filtfilt, butter

# Add ECGFounder to path
ECGFOUNDER_PATH = Path(__file__).parent.parent.parent / "github" / "ECGFounder"
sys.path.insert(0, str(ECGFOUNDER_PATH))

from net1d import Net1D

# Paths
MODEL_PATH = Path(__file__).parent.parent.parent / "github" / "1_lead_ECGFounder.pth"
TASKS_PATH = ECGFOUNDER_PATH / "tasks.txt"
OUTPUT_DIR = Path(__file__).parent / "output"

# Model parameters
TARGET_FS = 500  # ECGFounder expects 500 Hz
TARGET_LENGTH = 5000  # 10 seconds * 500 Hz
N_CLASSES = 150


def load_disease_labels():
    """Load the 150 disease class labels"""
    with open(TASKS_PATH, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def load_ecg_strips():
    """Load extracted ECG strips from .npy files"""
    print("1. Loading extracted ECG strips...")

    strips = []
    for i in range(1, 4):
        npy_path = OUTPUT_DIR / f"ecg_strip_{i}.npy"
        if npy_path.exists():
            strip = np.load(npy_path)
            strips.append(strip)
            print(f"   Strip {i}: {len(strip)} samples")
        else:
            print(f"   WARNING: {npy_path} not found")

    if not strips:
        raise FileNotFoundError("No ECG strips found in output/")

    return strips


def resample_to_500hz(signal, original_fs):
    """Resample signal from original_fs to 500 Hz"""
    target_samples = int(len(signal) * TARGET_FS / original_fs)
    resampled = resample(signal, target_samples)
    return resampled


def filter_bandpass(signal, fs):
    """
    ECGFounder preprocessing: bandpass filter + baseline removal
    Matches the preprocessing in ECGFounder/util.py
    """
    # Ensure 2D shape (1, samples)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    # 1. Remove power-line interference (50 Hz notch)
    b, a = iirnotch(50, 30, fs)
    filtered = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered[c] = filtfilt(b, a, signal[c])

    # 2. Bandpass filter (0.67 - 40 Hz)
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered[c] = filtfilt(b, a, filtered[c])

    # 3. Remove baseline wander (median filter)
    baseline = np.zeros_like(filtered)
    for c in range(filtered.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        baseline[c] = medfilt(filtered[c], kernel_size=kernel_size)

    filtered = filtered - baseline

    return filtered.squeeze()


def z_score_normalize(signal):
    """Z-score normalization"""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def preprocess_for_ecgfounder(strips, original_fs):
    """
    Full preprocessing pipeline for ECGFounder:
    1. Resample to 500 Hz
    2. Apply bandpass filter + baseline removal
    3. Z-score normalize
    4. Ensure correct length (5000 samples)
    """
    print("\n2. Resampling to 500 Hz...")
    print("3. Applying ECGFounder preprocessing...")

    processed_strips = []

    for i, strip in enumerate(strips):
        # Resample
        resampled = resample_to_500hz(strip, original_fs)
        print(f"   Strip {i+1}: {len(strip)} -> {len(resampled)} samples")

        # Bandpass filter + baseline removal
        filtered = filter_bandpass(resampled, TARGET_FS)

        # Z-score normalize
        normalized = z_score_normalize(filtered)

        # Ensure exact length (pad or truncate)
        if len(normalized) < TARGET_LENGTH:
            normalized = np.pad(normalized, (0, TARGET_LENGTH - len(normalized)), mode='constant')
        elif len(normalized) > TARGET_LENGTH:
            normalized = normalized[:TARGET_LENGTH]

        processed_strips.append(normalized)

    return processed_strips


def load_ecgfounder_model(device):
    """Load the 1-lead ECGFounder model"""
    print("\n4. Loading ECGFounder model...")

    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=N_CLASSES
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"   Model loaded from: {MODEL_PATH.name}")
    print(f"   Device: {device}")

    return model


def run_inference(model, processed_strips, device):
    """Run inference on processed ECG strips"""
    print("\n5. Running disease screening...")

    all_predictions = []

    with torch.no_grad():
        for i, strip in enumerate(processed_strips):
            # Prepare input: (batch=1, channels=1, length=5000)
            x = torch.FloatTensor(strip).unsqueeze(0).unsqueeze(0).to(device)

            # Forward pass
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()

            all_predictions.append(probs)
            print(f"   Strip {i+1}: inference complete")

    # Average predictions across strips
    avg_predictions = np.mean(all_predictions, axis=0)

    return avg_predictions, all_predictions


def display_results(predictions, labels, top_k=20, threshold=0.1):
    """Display screening results"""
    print("\n" + "=" * 70)
    print("ECG DISEASE SCREENING RESULTS")
    print("=" * 70)

    # Sort by probability
    sorted_indices = np.argsort(predictions)[::-1]

    print(f"\nTop {top_k} findings (probability > {threshold*100:.0f}%):\n")
    print(f"{'Rank':<6}{'Condition':<55}{'Prob':>8}")
    print("-" * 70)

    count = 0
    for idx in sorted_indices:
        prob = predictions[idx]
        if prob >= threshold and count < top_k:
            label = labels[idx]
            count += 1
            print(f"{count:<6}{label:<55}{prob*100:>7.1f}%")

    if count == 0:
        print("No conditions detected above threshold.")

    # Key clinical findings
    print("\n" + "-" * 70)
    print("KEY CLINICAL CATEGORIES:")
    print("-" * 70)

    clinical_groups = {
        "Rhythm": ["SINUS RHYTHM", "SINUS BRADYCARDIA", "SINUS TACHYCARDIA",
                   "ATRIAL FIBRILLATION", "ATRIAL FLUTTER"],
        "Conduction": ["RIGHT BUNDLE BRANCH BLOCK", "LEFT BUNDLE BRANCH BLOCK",
                       "LEFT ANTERIOR FASCICULAR BLOCK", "1ST DEGREE AV BLOCK"],
        "Ischemia/Infarct": ["ANTERIOR INFARCT", "INFERIOR INFARCT", "LATERAL INFARCT",
                            "SEPTAL INFARCT", "ACUTE MI"],
        "Hypertrophy": ["LEFT VENTRICULAR HYPERTROPHY", "RIGHT VENTRICULAR HYPERTROPHY",
                       "LEFT ATRIAL ENLARGEMENT", "RIGHT ATRIAL ENLARGEMENT"],
    }

    for group_name, conditions in clinical_groups.items():
        print(f"\n{group_name}:")
        for condition in conditions:
            for i, label in enumerate(labels):
                if condition in label.upper():
                    prob = predictions[i]
                    if prob > 0.05:
                        print(f"  - {labels[i]}: {prob*100:.1f}%")
                    break


def save_results(predictions, labels, output_path):
    """Save results to file"""
    with open(output_path, 'w') as f:
        f.write("ECG Disease Screening Results\n")
        f.write("=" * 50 + "\n\n")

        sorted_indices = np.argsort(predictions)[::-1]

        for idx in sorted_indices:
            prob = predictions[idx]
            label = labels[idx]
            f.write(f"{label}: {prob*100:.2f}%\n")

    print(f"\nResults saved to: {output_path}")


def main():
    print("=" * 70)
    print("ECG DISEASE SCREENING - ECGFounder (1-lead, 150 classes)")
    print("=" * 70)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 1. Load ECG strips
    strips = load_ecg_strips()

    # Calculate original sample rate (pixels / 10 seconds)
    original_fs = len(strips[0]) / 10.0
    print(f"   Original sample rate: {original_fs:.1f} Hz")

    # 2 & 3. Preprocess
    processed_strips = preprocess_for_ecgfounder(strips, original_fs)

    # 4. Load model
    model = load_ecgfounder_model(device)

    # 5. Run inference
    predictions, per_strip_predictions = run_inference(model, processed_strips, device)

    # Load disease labels
    labels = load_disease_labels()

    # Display results
    display_results(predictions, labels)

    # Save results
    save_results(predictions, labels, OUTPUT_DIR / "screening_results.txt")

    # Save numpy arrays
    np.save(OUTPUT_DIR / "predictions.npy", predictions)
    np.save(OUTPUT_DIR / "per_strip_predictions.npy", per_strip_predictions)

    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
