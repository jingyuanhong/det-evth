#!/usr/bin/env python3
"""
ECG Extraction and Disease Screening Pipeline

Supports both PDF and JPG/PNG image inputs.
Extracts ECG waveform and runs ECGFounder disease screening.
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.signal import butter, iirnotch, filtfilt, resample, medfilt

# Add ECGFounder to path
ECGFOUNDER_PATH = Path(__file__).parent.parent.parent / "github" / "ECGFounder"
sys.path.insert(0, str(ECGFOUNDER_PATH))

from net1d import Net1D

# Paths
MODEL_PATH = Path(__file__).parent.parent.parent / "github" / "1_lead_ECGFounder.pth"
TASKS_PATH = ECGFOUNDER_PATH / "tasks.txt"
OUTPUT_DIR = Path(__file__).parent / "output"

# Constants
TARGET_FS = 500
TARGET_LENGTH = 5000
N_CLASSES = 150


def load_image(file_path):
    """Load PDF or image file"""
    file_path = Path(file_path)
    print(f"Loading: {file_path.name}")

    if file_path.suffix.lower() == '.pdf':
        from pdf2image import convert_from_path
        images = convert_from_path(str(file_path), dpi=300)
        if not images:
            raise ValueError("Could not load PDF")
        image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")

    print(f"  Image size: {image.shape[1]}x{image.shape[0]} pixels")
    return image


def detect_waveform_pixels(image):
    """Detect red waveform pixels using BGR and HSV thresholds"""
    print("Detecting waveform pixels...")

    b, g, r = cv2.split(image)

    # BGR threshold for red color (relaxed for JPG compression artifacts)
    mask_bgr = ((b <= 180) & (g <= 180) & (r >= 150)).astype(np.uint8) * 255

    # HSV threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red hue wraps around, check both ends
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 50, 150]), np.array([10, 255, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([150, 50, 150]), np.array([180, 255, 255]))
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # Combine masks
    mask = cv2.bitwise_and(mask_bgr, mask_hsv)

    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_pixels = np.sum(mask > 0)
    print(f"  Found {num_pixels:,} waveform pixels")

    return mask


def find_strips(mask):
    """Find ECG strips by analyzing row-wise pixel distribution"""
    print("Finding strips...")

    height = mask.shape[0]
    pixels_per_row = np.sum(mask > 0, axis=1)

    # Smooth the distribution
    kernel_size = 5
    pixels_smoothed = np.convolve(pixels_per_row, np.ones(kernel_size)/kernel_size, mode='same')

    # Find rows with significant waveform content
    threshold = np.max(pixels_smoothed) * 0.1
    has_waveform = pixels_smoothed > threshold

    # Group consecutive rows into strips
    strips = []
    in_strip = False
    start = None
    min_gap = 30

    for i in range(height):
        if has_waveform[i]:
            if not in_strip:
                start = i
                in_strip = True
            last_row = i
        elif in_strip:
            if i - last_row > min_gap:
                if last_row - start > 30:
                    strips.append((start, last_row + 1))
                in_strip = False

    if in_strip and last_row - start > 30:
        strips.append((start, last_row + 1))

    print(f"  Found {len(strips)} strips")
    for i, (s, e) in enumerate(strips):
        print(f"    Strip {i+1}: rows {s}-{e} ({e-s} rows)")

    return strips


def extract_waveform(mask, y_start, y_end):
    """Extract waveform values from mask"""
    strip_mask = mask[y_start:y_end, :]
    height, width = strip_mask.shape

    waveform = np.full(width, np.nan)

    for x in range(width):
        col = strip_mask[:, x]
        white_pixels = np.where(col > 0)[0]
        if len(white_pixels) > 0:
            waveform[x] = np.mean(white_pixels)

    # Fill gaps with baseline
    valid = ~np.isnan(waveform)
    if np.sum(valid) > 0:
        baseline_y = np.median(waveform[valid])
        waveform[~valid] = baseline_y

    # Invert y-axis and convert to voltage
    waveform = height - waveform
    baseline = np.median(waveform)
    voltage = (waveform - baseline) / 175.0

    return voltage


def apply_extraction_filters(signal, fs):
    """Apply ECG filters during extraction"""
    pad_len = int(fs * 2)
    signal_padded = np.pad(signal, pad_len, mode='reflect')

    # High-pass 0.5 Hz
    b, a = butter(2, 0.5, btype='high', fs=fs)
    signal_padded = filtfilt(b, a, signal_padded)

    # Low-pass - use min of 40 Hz or Nyquist-1
    lowpass_freq = min(40, fs/2 - 1)
    b, a = butter(2, lowpass_freq, btype='low', fs=fs)
    signal_padded = filtfilt(b, a, signal_padded)

    # Notch 50 Hz only if sample rate allows
    if fs > 100:
        b, a = iirnotch(50, Q=30, fs=fs)
        signal_padded = filtfilt(b, a, signal_padded)

    return signal_padded[pad_len:-pad_len]


def preprocess_for_model(signal, original_fs):
    """Preprocess signal for ECGFounder model"""
    # Resample to 500 Hz
    target_samples = int(len(signal) * TARGET_FS / original_fs)
    resampled = resample(signal, target_samples)

    # Ensure 2D
    if resampled.ndim == 1:
        resampled = resampled.reshape(1, -1)

    # Bandpass filter (0.67-40 Hz)
    b, a = iirnotch(50, 30, TARGET_FS)
    filtered = filtfilt(b, a, resampled[0])

    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=TARGET_FS)
    filtered = filtfilt(b, a, filtered)

    # Baseline removal
    kernel_size = int(0.4 * TARGET_FS) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    baseline = medfilt(filtered, kernel_size=kernel_size)
    filtered = filtered - baseline

    # Z-score normalization
    normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)

    # Ensure correct length
    if len(normalized) < TARGET_LENGTH:
        normalized = np.pad(normalized, (0, TARGET_LENGTH - len(normalized)), mode='constant')
    elif len(normalized) > TARGET_LENGTH:
        normalized = normalized[:TARGET_LENGTH]

    return normalized


def load_model(device):
    """Load ECGFounder model"""
    print("\nLoading ECGFounder model...")

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
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    model.eval()

    print(f"  Model loaded, device: {device}")
    return model


def load_labels():
    """Load disease class labels"""
    with open(TASKS_PATH, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def run_inference(model, signals, device):
    """Run model inference"""
    print("\nRunning disease screening...")

    all_preds = []
    with torch.no_grad():
        for i, sig in enumerate(signals):
            x = torch.FloatTensor(sig).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            all_preds.append(probs)
            print(f"  Strip {i+1}: done")

    return np.array(all_preds)


def display_results(predictions, labels, filename):
    """Display screening results"""
    avg_pred = predictions.mean(axis=0)

    print("\n" + "=" * 70)
    print(f"SCREENING RESULTS: {filename}")
    print("=" * 70)

    # Table header
    print(f"\n| {'Condition':<40} | {'Strip1':>7} | {'Strip2':>7} | {'Strip3':>7} | {'Avg':>7} |")
    print("|" + "-"*42 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|")

    # Sort by average probability
    sorted_idx = np.argsort(avg_pred)[::-1]

    # Show top 20
    for rank, idx in enumerate(sorted_idx[:20], 1):
        name = labels[idx][:40]
        s1 = predictions[0, idx] * 100 if len(predictions) > 0 else 0
        s2 = predictions[1, idx] * 100 if len(predictions) > 1 else 0
        s3 = predictions[2, idx] * 100 if len(predictions) > 2 else 0
        avg = avg_pred[idx] * 100
        print(f"| {name:<40} | {s1:>6.1f}% | {s2:>6.1f}% | {s3:>6.1f}% | {avg:>6.1f}% |")

    return avg_pred


def save_results(predictions, avg_pred, labels, output_dir, filename):
    """Save results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    base_name = Path(filename).stem

    # Save predictions
    np.save(output_dir / f"{base_name}_predictions.npy", predictions)
    np.save(output_dir / f"{base_name}_avg_predictions.npy", avg_pred)

    # Save text results
    with open(output_dir / f"{base_name}_results.txt", 'w') as f:
        f.write(f"ECG Screening Results: {filename}\n")
        f.write("=" * 50 + "\n\n")

        sorted_idx = np.argsort(avg_pred)[::-1]
        for idx in sorted_idx:
            f.write(f"{labels[idx]}: {avg_pred[idx]*100:.2f}%\n")

    print(f"\nResults saved to: {output_dir}/{base_name}_*.npy/txt")


def plot_extracted_signals(signals, fs, output_dir, filename):
    """Plot extracted ECG signals"""
    fig, axes = plt.subplots(len(signals), 1, figsize=(16, 3*len(signals)))
    if len(signals) == 1:
        axes = [axes]

    for i, (sig, ax) in enumerate(zip(signals, axes)):
        time = np.arange(len(sig)) / fs
        ax.plot(time, sig, 'r-', linewidth=0.8)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'Strip {i+1}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)

    plt.tight_layout()
    base_name = Path(filename).stem
    plt.savefig(output_dir / f"{base_name}_extracted.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/{base_name}_extracted.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_and_screen.py <ecg.pdf|ecg.jpg>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        sys.exit(1)

    print("=" * 70)
    print("ECG EXTRACTION AND DISEASE SCREENING")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 1. Load image
    image = load_image(file_path)

    # 2. Detect waveform
    mask = detect_waveform_pixels(image)

    # 3. Find strips
    strips = find_strips(mask)

    if len(strips) == 0:
        print("ERROR: No strips found!")
        sys.exit(1)

    # 4. Extract waveforms
    print("\nExtracting waveforms...")
    signals_raw = []
    for i, (y_start, y_end) in enumerate(strips):
        sig = extract_waveform(mask, y_start, y_end)
        signals_raw.append(sig)
        print(f"  Strip {i+1}: {len(sig)} samples")

    # Calculate sample rate
    original_fs = len(signals_raw[0]) / 10.0
    print(f"  Sample rate: {original_fs:.1f} Hz")

    # 5. Apply extraction filters
    print("\nApplying filters...")
    signals_filtered = []
    for sig in signals_raw:
        filtered = apply_extraction_filters(sig, original_fs)
        signals_filtered.append(filtered)

    # 6. Plot extracted signals
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_extracted_signals(signals_filtered, original_fs, OUTPUT_DIR, file_path.name)

    # 7. Preprocess for model
    print("\nPreprocessing for model (resample to 500Hz, normalize)...")
    signals_preprocessed = []
    for sig in signals_filtered:
        preprocessed = preprocess_for_model(sig, original_fs)
        signals_preprocessed.append(preprocessed)

    # 8. Load model and run inference
    model = load_model(device)
    labels = load_labels()
    predictions = run_inference(model, signals_preprocessed, device)

    # 9. Display and save results
    avg_pred = display_results(predictions, labels, file_path.name)
    save_results(predictions, avg_pred, labels, OUTPUT_DIR, file_path.name)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
