#!/usr/bin/env python3
"""
Final Simple ECG Extractor - No Complex Thresholds!

Dead simple logic:
1. Find waveform pixels (color detection)
2. Any row with pixels = part of a strip
3. Group consecutive rows = strips
4. Extract, resample, done!
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate, signal as scipy_signal


def load_pdf(pdf_path):
    """Load PDF as image"""
    from pdf2image import convert_from_path

    print(f"Loading: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=300)

    if not images:
        raise ValueError("Could not load PDF")

    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    print(f"✓ Image: {image.shape[1]}x{image.shape[0]} pixels\n")
    return image


def detect_waveform_pixels(image):
    """
    Detect waveform pixels - strict thresholds, thin clean line.
    No morphological operations that would thicken the line.
    """
    print("Detecting waveform pixels...")

    b, g, r = cv2.split(image)

    # Strict BGR threshold
    mask_bgr = ((b <= 154) & (g <= 146) & (r >= 192) & (r <= 224)).astype(np.uint8) * 255

    # Strict HSV threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([157, 83, 192]), np.array([180, 255, 224]))

    # Combine (both must agree)
    mask = cv2.bitwise_and(mask_bgr, mask_hsv)

    num_pixels = np.sum(mask > 0)
    print(f"  Found {num_pixels:,} waveform pixels\n")

    return mask


def find_strips_simple(waveform_mask):
    """
    Dead simple strip detection:
    - Any row with waveform pixels = part of a strip
    - Group consecutive rows = one strip
    """
    print("Finding strips (simple method)...")

    height = waveform_mask.shape[0]

    # Count pixels in each row
    pixels_per_row = np.sum(waveform_mask > 0, axis=1)

    # Any row with ANY pixels is part of a strip
    has_waveform = pixels_per_row > 0

    print(f"  {np.sum(has_waveform)} rows have waveform pixels")

    # Find consecutive groups
    strips = []
    in_strip = False
    start = None
    min_gap = 20  # Allow max 20-row gaps within a strip

    for i in range(height):
        if has_waveform[i]:
            if not in_strip:
                start = i
                in_strip = True
            last_pixel_row = i
        elif in_strip:
            # Check if gap is too large
            if i - last_pixel_row > min_gap:
                # End this strip
                if last_pixel_row - start > 50:  # Min 50 rows for valid strip
                    strips.append((start, last_pixel_row + 1))
                in_strip = False
                start = None

    # Handle last strip
    if in_strip and last_pixel_row - start > 50:
        strips.append((start, last_pixel_row + 1))

    print(f"\n  Detected {len(strips)} strips:")
    for i, (start, end) in enumerate(strips):
        px_count = np.sum(pixels_per_row[start:end])
        print(f"    Strip {i+1}: rows {start}-{end} ({end-start} rows, {px_count:,} pixels)")

    print()

    # Visualize
    visualize_strip_detection(waveform_mask, strips, pixels_per_row)

    return strips


def visualize_strip_detection(mask, strips, pixels_per_row):
    """Create diagnostic visualization"""
    Path("output/debug").mkdir(parents=True, exist_ok=True)

    height, width = mask.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Plot 1: Pixels per row
    ax1.barh(range(height), pixels_per_row, height=1, color='steelblue', alpha=0.7)

    # Mark detected strips
    for i, (start, end) in enumerate(strips):
        ax1.axhline(start, color='green', linewidth=2, label=f'Strip {i+1} start' if i == 0 else '')
        ax1.axhline(end, color='red', linewidth=2, label=f'Strip end' if i == 0 else '')
        # Shade strip region
        ax1.axhspan(start, end, alpha=0.2, color='yellow')

    ax1.set_xlabel('Waveform Pixels per Row', fontsize=11)
    ax1.set_ylabel('Row Number', fontsize=11)
    ax1.set_title('Waveform Distribution & Detected Strips', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Mask visualization
    ax2.imshow(mask, cmap='hot', aspect='auto')

    # Draw strip boundaries
    for i, (start, end) in enumerate(strips):
        ax2.axhline(start, color='lime', linewidth=3)
        ax2.axhline(end, color='red', linewidth=3)
        ax2.text(10, (start+end)//2, f'Strip {i+1}',
                color='white', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax2.set_xlabel('Column', fontsize=11)
    ax2.set_ylabel('Row', fontsize=11)
    ax2.set_title(f'Waveform Mask ({len(strips)} strips detected)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/debug/strip_detection_final.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: output/debug/strip_detection_final.png")

    plt.close()


def extract_waveform(mask, y_start, y_end, strip_num):
    """
    Extract waveform from thin mask.
    For gaps (baseline areas with no pixels), fill with baseline y-value.
    """
    print(f"Extracting strip {strip_num}...")

    strip_mask = mask[y_start:y_end, :]
    height, width = strip_mask.shape

    # Save mask
    cv2.imwrite(f'output/debug/mask_strip_{strip_num}.png', strip_mask)

    # Extract y-position for each column
    waveform = np.full(width, np.nan)

    for x in range(width):
        col = strip_mask[:, x]
        white_pixels = np.where(col > 0)[0]

        if len(white_pixels) > 0:
            waveform[x] = np.mean(white_pixels)

    # Find baseline y (median of valid values)
    valid = ~np.isnan(waveform)
    valid_count = np.sum(valid)
    print(f"  Valid columns: {valid_count}/{width} ({valid_count/width*100:.1f}%)")

    if valid_count > 0:
        baseline_y = np.median(waveform[valid])

        # Fill gaps with baseline (flat line in gap regions)
        waveform[~valid] = baseline_y

    # Invert y-axis
    waveform = height - waveform

    # Convert to mV
    baseline = np.median(waveform)
    voltage = (waveform - baseline) / 175.0

    print(f"  ✓ {len(voltage)} samples, range [{voltage.min():.3f}, {voltage.max():.3f}] mV")

    return voltage


def concatenate_signals(strips_signals):
    """Simply concatenate the extracted signals - no extra processing"""
    print("\nConcatenating strips...")

    # Direct concatenation
    full_signal = np.concatenate(strips_signals)

    # Calculate sample rate: 3300 pixels / 10 seconds = 330 Hz per strip
    fs = len(strips_signals[0]) / 10.0
    time = np.arange(len(full_signal)) / fs

    print(f"  Total: {len(full_signal)} samples ({time[-1]:.1f}s)")
    print(f"  Sample rate: {fs:.1f} Hz")

    return full_signal, time, fs


def apply_filters(signal, fs=512):
    """
    Apply ECG filtering with ZERO PHASE DISTORTION

    Uses filtfilt (forward-backward filtering) to eliminate phase shift
    that was causing shadow/ghost signals in the output
    """

    # Design filters (use 'ba' format for filtfilt compatibility)
    # High-pass: remove baseline wander (< 0.5 Hz)
    b_hp, a_hp = scipy_signal.butter(2, 0.5, 'highpass', fs=fs)

    # Low-pass: remove high-frequency noise (> 40 Hz)
    b_lp, a_lp = scipy_signal.butter(2, 40, 'lowpass', fs=fs)

    # Apply zero-phase filtering (forward-backward)
    # This eliminates the phase distortion that was creating artifacts
    signal_hp = scipy_signal.filtfilt(b_hp, a_hp, signal)
    signal_filtered = scipy_signal.filtfilt(b_lp, a_lp, signal_hp)

    # NO median filter - it was introducing artifacts
    # The zero-phase filtering is sufficient

    return signal_filtered


def estimate_hr(signal, fs=512):
    """Estimate heart rate"""
    from scipy.signal import find_peaks

    threshold = np.mean(signal) + 0.5 * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=fs*0.4)

    if len(peaks) < 2:
        return None

    duration = len(signal) / fs
    hr = (len(peaks) / duration) * 60
    return hr


def plot_three_strips(signals, fs):
    """Plot three separate 10-second ECG strips"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    for i, (sig, ax) in enumerate(zip(signals, axes)):
        time = np.arange(len(sig)) / fs
        ax.plot(time, sig, 'r-', linewidth=0.8)
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Voltage (mV)', fontsize=11)
        ax.set_title(f'Strip {i+1}: 0-10 seconds', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig('output/ecg_final.png', dpi=150, bbox_inches='tight')
    print("  output/ecg_final.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ecg_final.py <ecg.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    print("=" * 70)
    print("ECG EXTRACTOR - Three 10-second strips")
    print("=" * 70)
    print()

    # 1. Load PDF
    image = load_pdf(pdf_path)

    # 2. Detect waveform pixels
    mask = detect_waveform_pixels(image)

    # 3. Find strips
    strips = find_strips_simple(mask)

    if len(strips) == 0:
        print("\n❌ ERROR: No strips found!")
        sys.exit(1)

    if len(strips) != 3:
        print(f"\n⚠ WARNING: Found {len(strips)} strips (expected 3)")

    # 4. Extract waveform from each strip separately
    print("\nExtracting waveforms...")
    signals = []
    for i, (y_start, y_end) in enumerate(strips):
        sig = extract_waveform(mask, y_start, y_end, i+1)
        signals.append(sig)

    # Calculate sample rate (pixels / 10 seconds)
    fs = len(signals[0]) / 10.0

    # 5. Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for i, sig in enumerate(signals):
        print(f"\nStrip {i+1}: {len(sig)} samples, [{sig.min():.3f}, {sig.max():.3f}] mV")

    print(f"\nSample rate: {fs:.1f} Hz")
    print(f"Duration: 10 seconds per strip")

    # 6. Save each strip separately
    print("\nSaving...")
    Path("output").mkdir(exist_ok=True)
    for i, sig in enumerate(signals):
        np.save(f'output/ecg_strip_{i+1}.npy', sig)
        print(f"  output/ecg_strip_{i+1}.npy")

    # 7. Plot three strips
    print("\nPlotting...")
    plot_three_strips(signals, fs)

    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
