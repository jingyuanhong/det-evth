#!/usr/bin/env python3
"""
Simple and robust Apple Watch ECG extractor

Uses red pixel distribution directly to find strips.
No complex thresholds - just find where the red pixels are!
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

    pil_image = images[0]
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    print(f"✓ Image: {image.shape[1]}x{image.shape[0]} pixels\n")
    return image


def find_red_pixels_simple(image):
    """Simple red pixel detection"""
    b, g, r = cv2.split(image)

    # Red is dominant: R > 100 and R > G+30 and R > B+30
    red_mask = ((r > 100) & (r > g + 30) & (r > b + 30)).astype(np.uint8) * 255

    # Count red pixels
    num_red = np.sum(red_mask > 0)
    print(f"Found {num_red:,} red pixels ({num_red/(image.shape[0]*image.shape[1])*100:.2f}% of image)")

    return red_mask


def detect_strips_from_red(image, red_mask):
    """
    Detect strips using red pixel horizontal projection
    This is foolproof: strips are where red pixels are!
    """
    height, width = image.shape[:2]

    # Count red pixels in each row
    red_per_row = np.sum(red_mask > 0, axis=1)

    # Smooth to reduce noise
    from scipy.ndimage import gaussian_filter1d
    red_per_row_smooth = gaussian_filter1d(red_per_row.astype(float), sigma=10)

    # Find threshold: rows with significant red content
    # Use mean + 0.5*std as threshold
    threshold = np.mean(red_per_row_smooth) + 0.5 * np.std(red_per_row_smooth)

    print(f"Red pixels per row: mean={np.mean(red_per_row):.1f}, max={np.max(red_per_row)}, threshold={threshold:.1f}")

    # Find continuous regions above threshold
    above_threshold = red_per_row_smooth > threshold

    regions = []
    in_region = False
    start = None
    min_height = height * 0.08  # At least 8% of image height

    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_region:
            start = i
            in_region = True
        elif not above_threshold[i] and in_region:
            if i - start > min_height:
                regions.append((start, i))
            in_region = False
            start = None

    # Handle last region
    if in_region and len(above_threshold) - start > min_height:
        regions.append((start, len(above_threshold)))

    print(f"\nDetected {len(regions)} ECG strips:")
    for i, (start, end) in enumerate(regions):
        print(f"  Strip {i+1}: rows {start}-{end} (height: {end-start} px)")

    # Save diagnostic plot
    Path("output/debug").mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Plot projection
    ax1.plot(red_per_row_smooth, range(height))
    ax1.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.1f}')
    ax1.set_xlabel('Red Pixels per Row (smoothed)')
    ax1.set_ylabel('Row Number')
    ax1.set_title('Horizontal Projection')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show detected regions on image
    img_copy = image.copy()
    for start, end in regions:
        cv2.line(img_copy, (0, start), (width, start), (0, 255, 0), 5)
        cv2.line(img_copy, (0, end), (width, end), (0, 0, 255), 5)

    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_rgb)
    ax2.set_title(f'Detected {len(regions)} Strips')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('output/debug/strip_detection.png', dpi=150, bbox_inches='tight')
    print("  Saved: output/debug/strip_detection.png\n")
    plt.close()

    return regions


def extract_waveform_from_strip(image, y_start, y_end, strip_index):
    """Extract waveform from one strip"""
    strip = image[y_start:y_end, :]
    height, width = strip.shape[:2]

    print(f"Extracting strip {strip_index}...")

    # Find red pixels
    b, g, r = cv2.split(strip)
    red_mask = ((r > 100) & (r > g + 30) & (r > b + 30)).astype(np.uint8) * 255

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.medianBlur(red_mask, 3)

    # Save mask
    cv2.imwrite(f'output/debug/mask_strip_{strip_index}.png', red_mask)

    # Extract waveform column by column
    waveform = []
    for x in range(width):
        column = red_mask[:, x]
        red_y = np.where(column > 0)[0]

        if len(red_y) > 0:
            # Use median for robustness
            y_pos = np.median(red_y)
        else:
            y_pos = np.nan

        waveform.append(y_pos)

    waveform = np.array(waveform)

    # Interpolate gaps
    nans = np.isnan(waveform)
    if nans.any() and not nans.all():
        x_idx = np.arange(len(waveform))
        waveform[nans] = np.interp(x_idx[nans], x_idx[~nans], waveform[~nans])

    # Invert (image coords to signal)
    waveform = height - waveform

    # Remove outliers
    median = np.median(waveform)
    mad = np.median(np.abs(waveform - median))
    if mad > 0:
        outliers = np.abs(waveform - median) > (6 * mad)
        waveform[outliers] = median

    print(f"  ✓ {len(waveform)} samples, {np.sum(nans)} gaps filled")

    return waveform


def pixels_to_voltage(waveform, strip_height):
    """Convert pixel coordinates to voltage (mV)"""
    # Normalize to baseline
    baseline = np.median(waveform)
    pixels_from_baseline = waveform - baseline

    # Estimate: ~15-20 pixels per mm at 300 DPI
    # 10 mm = 1 mV (standard ECG)
    pixels_per_mm = 18  # Calibrated for 300 DPI

    mm = pixels_from_baseline / pixels_per_mm
    voltage = mm / 10.0  # 10 mm/mV

    return voltage


def resample_strip(signal, target_samples=5120):
    """Resample one strip to exactly 5120 samples (10s at 512Hz)"""
    duration = 10.0
    x_old = np.linspace(0, duration, len(signal))
    x_new = np.linspace(0, duration, target_samples)

    f = interpolate.interp1d(x_old, signal, kind='linear')
    return f(x_new)


def apply_filters(signal, fs=512):
    """Apply basic filtering"""
    # High-pass: remove baseline drift (0.5 Hz)
    sos_hp = scipy_signal.butter(2, 0.5, 'highpass', fs=fs, output='sos')
    signal = scipy_signal.sosfilt(sos_hp, signal)

    # Low-pass: remove high-freq noise (40 Hz)
    sos_lp = scipy_signal.butter(2, 40, 'lowpass', fs=fs, output='sos')
    signal = scipy_signal.sosfilt(sos_lp, signal)

    return signal


def estimate_heart_rate(signal, fs=512):
    """Estimate heart rate from peaks"""
    # Find peaks
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, height=threshold, distance=fs*0.4)

    if len(peaks) < 2:
        return None

    # Calculate rate
    duration = len(signal) / fs
    hr = (len(peaks) / duration) * 60

    return hr


def plot_results(signal, time):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 12))

    # 1. Full signal
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(time, signal, 'r-', linewidth=0.8)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Voltage (mV)', fontsize=11)
    ax1.set_title(f'Complete ECG Signal - {time[-1]:.1f} seconds', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2-4. Three 10-second strips
    for i in range(3):
        ax = plt.subplot(4, 3, 4+i)
        mask = (time >= i*10) & (time < (i+1)*10)
        if np.any(mask):
            ax.plot(time[mask], signal[mask], 'r-', linewidth=1.0)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Voltage (mV)', fontsize=10)
            ax.set_title(f'Strip {i+1}: {i*10}-{(i+1)*10}s', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # 5. Zoomed first 3 seconds
    ax5 = plt.subplot(4, 3, 7)
    mask = time <= 3
    if np.any(mask):
        ax5.plot(time[mask], signal[mask], 'r-', linewidth=1.2)
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.set_ylabel('Voltage (mV)', fontsize=10)
        ax5.set_title('First 3 seconds (zoomed)', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # 6. Histogram
    ax6 = plt.subplot(4, 3, 8)
    ax6.hist(signal, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Voltage (mV)', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Voltage Distribution', fontsize=11, fontweight='bold')
    ax6.axvline(signal.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {signal.mean():.3f}')
    ax6.legend(fontsize=9)

    # 7. Power spectrum
    ax7 = plt.subplot(4, 3, 9)
    freqs = np.fft.rfftfreq(len(signal), 1/512)
    fft = np.abs(np.fft.rfft(signal))
    ax7.semilogy(freqs, fft)
    ax7.set_xlabel('Frequency (Hz)', fontsize=10)
    ax7.set_ylabel('Magnitude (log)', fontsize=10)
    ax7.set_title('Frequency Spectrum', fontsize=11, fontweight='bold')
    ax7.set_xlim(0, 50)
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/ecg_final.png', dpi=150, bbox_inches='tight')
    print("  output/ecg_final.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ecg_simple.py <ecg.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("APPLE WATCH ECG EXTRACTOR")
    print("=" * 70)
    print()

    # 1. Load image
    image = load_pdf(pdf_path)

    # 2. Find red pixels
    print("Finding red pixels...")
    red_mask = find_red_pixels_simple(image)
    print()

    # 3. Detect strips from red pixel distribution
    print("Detecting strips...")
    regions = detect_strips_from_red(image, red_mask)

    if len(regions) == 0:
        print("\n❌ ERROR: No ECG strips detected!")
        print("\nPossible issues:")
        print("  1. PDF doesn't have red waveform")
        print("  2. Color thresholds need adjustment")
        print("  3. Different PDF format than expected")
        print("\nCheck output/debug/strip_detection.png for visualization")
        sys.exit(1)

    if len(regions) != 3:
        print(f"\n⚠ WARNING: Expected 3 strips, found {len(regions)}")
        print("Continuing anyway...\n")

    # 4. Extract waveforms
    print("Extracting waveforms...")
    signals_raw = []

    for i, (y_start, y_end) in enumerate(regions):
        waveform = extract_waveform_from_strip(image, y_start, y_end, i+1)
        voltage = pixels_to_voltage(waveform, y_end - y_start)
        signals_raw.append(voltage)

    print()

    # 5. Resample to 512 Hz
    print("Resampling to 512 Hz...")
    signals_resampled = []
    for i, sig in enumerate(signals_raw):
        resampled = resample_strip(sig, target_samples=5120)
        signals_resampled.append(resampled)
        print(f"  Strip {i+1}: {len(sig)} → {len(resampled)} samples")

    # 6. Concatenate
    full_signal = np.concatenate(signals_resampled)
    time = np.arange(len(full_signal)) / 512.0
    print(f"  Total: {len(full_signal)} samples ({time[-1]:.1f}s)\n")

    # 7. Filter
    print("Applying filters...")
    full_signal = apply_filters(full_signal)
    print("  ✓ High-pass (0.5 Hz) + Low-pass (40 Hz)\n")

    # 8. Analyze
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nSignal:")
    print(f"  Samples: {len(full_signal):,}")
    print(f"  Duration: {time[-1]:.2f} seconds")
    print(f"  Sampling rate: 512 Hz")
    print(f"  Voltage range: [{full_signal.min():.3f}, {full_signal.max():.3f}] mV")
    print(f"  Mean: {full_signal.mean():.3f} mV")
    print(f"  Std dev: {full_signal.std():.3f} mV")

    hr = estimate_heart_rate(full_signal)
    if hr:
        print(f"\nEstimated heart rate: {hr:.1f} bpm")

    # 9. Save
    print("\nSaving...")
    Path("output").mkdir(exist_ok=True)
    np.save('output/extracted_ecg_signal.npy', full_signal)
    np.save('output/extracted_ecg_time.npy', time)
    print("  output/extracted_ecg_signal.npy")
    print("  output/extracted_ecg_time.npy")

    # 10. Plot
    print("\nPlotting...")
    plot_results(full_signal, time)

    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  output/ecg_final.png - Comprehensive visualization")
    print("  output/debug/strip_detection.png - Strip detection analysis")
    print("  output/debug/mask_strip_*.png - Red pixel masks")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
