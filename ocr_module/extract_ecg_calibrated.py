#!/usr/bin/env python3
"""
Calibrated Apple Watch ECG Extractor

Uses exact color thresholds determined from color analysis.
Waveform color: R=208, G=38, B=57 (pink/coral, NOT pure red!)
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate, signal as scipy_signal, ndimage


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


def find_waveform_pixels_calibrated(image):
    """
    Find ECG waveform pixels using calibrated color thresholds
    Based on color analysis: R=208, G=38, B=57
    """
    print("Detecting waveform pixels with calibrated thresholds...")

    b, g, r = cv2.split(image)

    # Method 1: BGR threshold (from color analysis)
    # R: 192-224, G: 0-146, B: 0-154
    mask_bgr = ((b >= 0) & (b <= 154) &
                (g >= 0) & (g <= 146) &
                (r >= 192) & (r <= 224)).astype(np.uint8) * 255

    # Method 2: HSV threshold (from color analysis)
    # H: 157-180, S: 83-255, V: 192-224
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([157, 83, 192])
    upper_hsv = np.array([180, 255, 224])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine both methods (both must agree)
    mask = cv2.bitwise_and(mask_bgr, mask_hsv)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_pixels = np.sum(mask > 0)
    print(f"  Found {num_pixels:,} waveform pixels ({num_pixels/(image.shape[0]*image.shape[1])*100:.3f}% of image)")
    print(f"  BGR mask: {np.sum(mask_bgr>0):,} pixels")
    print(f"  HSV mask: {np.sum(mask_hsv>0):,} pixels")
    print(f"  Combined: {num_pixels:,} pixels\n")

    return mask


def detect_strips_robust(image, waveform_mask):
    """
    Detect strips using red pixel horizontal projection
    """
    print("Detecting ECG strips...")
    height, width = image.shape[:2]

    # Count waveform pixels in each row
    pixels_per_row = np.sum(waveform_mask > 0, axis=1).astype(float)

    # Smooth with Gaussian filter (sigma=15 for robustness)
    pixels_per_row_smooth = ndimage.gaussian_filter1d(pixels_per_row, sigma=15)

    # Adaptive threshold: mean + 0.3*std (lower threshold to catch all strips)
    mean_val = np.mean(pixels_per_row_smooth[pixels_per_row_smooth > 0])
    std_val = np.std(pixels_per_row_smooth[pixels_per_row_smooth > 0])
    threshold = mean_val + 0.3 * std_val

    print(f"  Pixels per row: mean={mean_val:.1f}, std={std_val:.1f}, threshold={threshold:.1f}")

    # Find regions above threshold
    above_threshold = pixels_per_row_smooth > threshold

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

    print(f"\n  Detected {len(regions)} ECG strips:")
    for i, (start, end) in enumerate(regions):
        print(f"    Strip {i+1}: rows {start}-{end} (height: {end-start} px)")

    # Save diagnostic visualization
    Path("output/debug").mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Projection plot
    ax1.plot(pixels_per_row_smooth, range(height))
    ax1.axvline(threshold, color='r', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold:.1f}')
    ax1.set_xlabel('Waveform Pixels per Row (smoothed)', fontsize=11)
    ax1.set_ylabel('Row Number', fontsize=11)
    ax1.set_title('Horizontal Projection', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Image with detected regions
    img_vis = image.copy()
    for start, end in regions:
        cv2.line(img_vis, (0, start), (width, start), (0, 255, 0), 5)
        cv2.line(img_vis, (0, end), (width, end), (0, 0, 255), 5)
        cv2.putText(img_vis, f'Strip {regions.index((start,end))+1} Start',
                    (10, start-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_vis, f'End',
                    (10, end+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_rgb)
    ax2.set_title(f'Detected {len(regions)} Strips', fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('output/debug/strip_detection_calibrated.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: output/debug/strip_detection_calibrated.png\n")
    plt.close()

    return regions


def extract_waveform_from_strip(image, y_start, y_end, strip_index, waveform_mask):
    """Extract waveform from one strip"""
    strip = image[y_start:y_end, :]
    mask = waveform_mask[y_start:y_end, :]
    height, width = strip.shape[:2]

    print(f"Extracting waveform from strip {strip_index}...")

    # Save mask for debugging
    cv2.imwrite(f'output/debug/mask_strip_{strip_index}.png', mask)

    # Extract waveform column by column
    waveform = []
    missing_count = 0

    for x in range(width):
        column = mask[:, x]
        waveform_y = np.where(column > 0)[0]

        if len(waveform_y) > 0:
            # Use median for robustness against outliers
            y_pos = np.median(waveform_y)
        else:
            y_pos = np.nan
            missing_count += 1

        waveform.append(y_pos)

    waveform = np.array(waveform)

    # Interpolate missing values
    nans = np.isnan(waveform)
    if nans.any() and not nans.all():
        x_idx = np.arange(len(waveform))
        waveform[nans] = np.interp(x_idx[nans], x_idx[~nans], waveform[~nans])

    # Invert y-axis (image coords → voltage coords)
    waveform = height - waveform

    # Remove extreme outliers using MAD
    median = np.median(waveform)
    mad = np.median(np.abs(waveform - median))
    if mad > 0:
        outliers = np.abs(waveform - median) > (5 * mad)
        num_outliers = np.sum(outliers)
        if num_outliers > 0:
            waveform[outliers] = median
            print(f"  Removed {num_outliers} outliers")

    print(f"  ✓ {len(waveform)} samples extracted, {missing_count} gaps interpolated")

    return waveform


def pixels_to_voltage(waveform, strip_height):
    """Convert pixel coordinates to voltage (mV)"""
    # Normalize to baseline
    baseline = np.median(waveform)
    pixels_from_baseline = waveform - baseline

    # Calibration: ~17-18 pixels per mm at 300 DPI
    # 10 mm = 1 mV (standard ECG calibration)
    pixels_per_mm = 17.5

    mm_from_baseline = pixels_from_baseline / pixels_per_mm
    voltage_mv = mm_from_baseline / 10.0

    return voltage_mv


def resample_strip(signal, target_samples=5120):
    """Resample one 10-second strip to exactly 5120 samples (512 Hz)"""
    duration = 10.0
    x_old = np.linspace(0, duration, len(signal))
    x_new = np.linspace(0, duration, target_samples)

    f = interpolate.interp1d(x_old, signal, kind='linear', fill_value='extrapolate')
    return f(x_new)


def apply_filters(signal, fs=512):
    """Apply ECG filtering"""
    # High-pass: remove baseline drift (0.5 Hz)
    sos_hp = scipy_signal.butter(2, 0.5, 'highpass', fs=fs, output='sos')
    signal_filtered = scipy_signal.sosfilt(sos_hp, signal)

    # Low-pass: remove high-frequency noise (40 Hz)
    sos_lp = scipy_signal.butter(2, 40, 'lowpass', fs=fs, output='sos')
    signal_filtered = scipy_signal.sosfilt(sos_lp, signal_filtered)

    return signal_filtered


def estimate_heart_rate(signal, fs=512):
    """Estimate heart rate from R-peaks"""
    from scipy.signal import find_peaks

    # Find peaks
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=fs*0.4)

    if len(peaks) < 2:
        return None

    duration = len(signal) / fs
    hr = (len(peaks) / duration) * 60

    return hr


def plot_comprehensive_results(signal, time):
    """Create detailed visualization"""
    fig = plt.figure(figsize=(18, 12))

    # 1. Full 30s signal
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(time, signal, 'r-', linewidth=0.7)
    ax1.set_xlabel('Time (seconds)', fontsize=10)
    ax1.set_ylabel('Voltage (mV)', fontsize=10)
    ax1.set_title(f'Complete ECG Signal ({time[-1]:.1f} seconds)',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark 10-second boundaries
    for i in [10, 20]:
        ax1.axvline(i, color='gray', linestyle='--', alpha=0.5)

    # 2-4. Individual 10s strips
    for i in range(3):
        ax = plt.subplot(4, 3, 4+i)
        mask = (time >= i*10) & (time < (i+1)*10)
        if np.any(mask):
            ax.plot(time[mask], signal[mask], 'r-', linewidth=0.8)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Voltage (mV)', fontsize=9)
            ax.set_title(f'Strip {i+1}: {i*10}-{(i+1)*10}s',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # 5. Zoomed view (first 3 seconds)
    ax5 = plt.subplot(4, 2, 4)
    mask = time <= 3
    if np.any(mask):
        ax5.plot(time[mask], signal[mask], 'r-', linewidth=1.2)
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.set_ylabel('Voltage (mV)', fontsize=10)
        ax5.set_title('First 3 Seconds (Zoomed)', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # 6. Histogram
    ax6 = plt.subplot(4, 2, 6)
    ax6.hist(signal, bins=60, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Voltage (mV)', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Voltage Distribution', fontsize=11, fontweight='bold')
    ax6.axvline(signal.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {signal.mean():.3f} mV')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # 7. Power spectrum
    ax7 = plt.subplot(4, 2, 7)
    freqs = np.fft.rfftfreq(len(signal), 1/512)
    fft = np.abs(np.fft.rfft(signal))
    ax7.semilogy(freqs, fft, linewidth=0.8)
    ax7.set_xlabel('Frequency (Hz)', fontsize=10)
    ax7.set_ylabel('Magnitude (log)', fontsize=10)
    ax7.set_title('Frequency Spectrum', fontsize=11, fontweight='bold')
    ax7.set_xlim(0, 50)
    ax7.grid(True, alpha=0.3)

    # 8. Statistics box
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')

    hr = estimate_heart_rate(signal)

    stats_text = f"""
ECG SIGNAL STATISTICS

Duration: {time[-1]:.2f} seconds
Samples: {len(signal):,}
Sampling Rate: 512 Hz

Voltage Statistics:
  Mean: {signal.mean():.3f} mV
  Std Dev: {signal.std():.3f} mV
  Range: [{signal.min():.3f}, {signal.max():.3f}] mV
  Median: {np.median(signal):.3f} mV

Heart Rate:
  Estimated: {hr:.1f} bpm
""" if hr else f"""
ECG SIGNAL STATISTICS

Duration: {time[-1]:.2f} seconds
Samples: {len(signal):,}
Sampling Rate: 512 Hz

Voltage Statistics:
  Mean: {signal.mean():.3f} mV
  Std Dev: {signal.std():.3f} mV
  Range: [{signal.min():.3f}, {signal.max():.3f}] mV
  Median: {np.median(signal):.3f} mV

Heart Rate:
  Could not estimate
"""

    ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig('output/ecg_calibrated_final.png', dpi=150, bbox_inches='tight')
    print("  output/ecg_calibrated_final.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ecg_calibrated.py <ecg.pdf>")
        print("\nThis version uses calibrated color thresholds from color analysis.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("CALIBRATED APPLE WATCH ECG EXTRACTOR")
    print("=" * 70)
    print("Using color thresholds: R=192-224, G=0-146, B=0-154")
    print("=" * 70)
    print()

    # 1. Load image
    image = load_pdf(pdf_path)

    # 2. Find waveform pixels with calibrated thresholds
    waveform_mask = find_waveform_pixels_calibrated(image)

    # 3. Detect strips
    regions = detect_strips_robust(image, waveform_mask)

    if len(regions) == 0:
        print("\n❌ ERROR: No ECG strips detected!")
        print("Check output/debug/strip_detection_calibrated.png")
        sys.exit(1)

    if len(regions) != 3:
        print(f"\n⚠ WARNING: Expected 3 strips, found {len(regions)}")
        print("Continuing with detected strips...\n")

    # 4. Extract waveforms
    print("Extracting waveforms...")
    signals_raw = []

    for i, (y_start, y_end) in enumerate(regions):
        waveform = extract_waveform_from_strip(image, y_start, y_end, i+1, waveform_mask)
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

    # 8. Results
    print("=" * 70)
    print("EXTRACTION RESULTS")
    print("=" * 70)
    print(f"\nSignal Properties:")
    print(f"  Samples: {len(full_signal):,}")
    print(f"  Duration: {time[-1]:.2f} seconds")
    print(f"  Sampling Rate: 512 Hz")
    print(f"  Voltage Range: [{full_signal.min():.3f}, {full_signal.max():.3f}] mV")
    print(f"  Mean: {full_signal.mean():.3f} mV")
    print(f"  Std Dev: {full_signal.std():.3f} mV")

    hr = estimate_heart_rate(full_signal)
    if hr:
        print(f"\nEstimated Heart Rate: {hr:.1f} bpm")

    # 9. Save
    print("\nSaving results...")
    Path("output").mkdir(exist_ok=True)
    np.save('output/extracted_ecg_signal.npy', full_signal)
    np.save('output/extracted_ecg_time.npy', time)
    print("  output/extracted_ecg_signal.npy")
    print("  output/extracted_ecg_time.npy")

    # 10. Plot
    print("\nCreating visualization...")
    plot_comprehensive_results(full_signal, time)

    print("\n" + "=" * 70)
    print("✓ EXTRACTION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  output/ecg_calibrated_final.png")
    print("  output/debug/strip_detection_calibrated.png")
    print("  output/debug/mask_strip_*.png")
    print("  output/debug/color_analysis.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
