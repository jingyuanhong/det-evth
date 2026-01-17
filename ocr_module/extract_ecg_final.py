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
    Detect waveform pixels using calibrated thresholds
    BGR: R∈[192,224], G∈[0,146], B∈[0,154]
    HSV: H∈[157,180], S∈[83,255], V∈[192,224]
    """
    print("Detecting waveform pixels...")

    b, g, r = cv2.split(image)

    # BGR threshold
    mask_bgr = ((b <= 154) & (g <= 146) & (r >= 192) & (r <= 224)).astype(np.uint8) * 255

    # HSV threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([157, 83, 192]), np.array([180, 255, 224]))

    # Combine (both must agree)
    mask = cv2.bitwise_and(mask_bgr, mask_hsv)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
    """Extract waveform from strip"""
    print(f"Extracting strip {strip_num}...")

    strip_mask = mask[y_start:y_end, :]
    height, width = strip_mask.shape

    # Save mask
    cv2.imwrite(f'output/debug/mask_strip_{strip_num}.png', strip_mask)

    # Extract column by column
    waveform = []

    for x in range(width):
        col = strip_mask[:, x]
        wf_pixels = np.where(col > 0)[0]

        if len(wf_pixels) > 0:
            y_pos = np.median(wf_pixels)  # Median for robustness
        else:
            y_pos = np.nan

        waveform.append(y_pos)

    waveform = np.array(waveform)

    # Interpolate gaps
    nans = np.isnan(waveform)
    if nans.any() and not nans.all():
        x_idx = np.arange(len(waveform))
        waveform[nans] = np.interp(x_idx[nans], x_idx[~nans], waveform[~nans])

    # Invert (image coords → signal)
    waveform = height - waveform

    # Remove extreme outliers
    median = np.median(waveform)
    mad = np.median(np.abs(waveform - median))
    if mad > 0:
        outliers = np.abs(waveform - median) > (5 * mad)
        if np.any(outliers):
            waveform[outliers] = median
            print(f"  Removed {np.sum(outliers)} outliers")

    # Convert to voltage (pixels → mV)
    baseline = np.median(waveform)
    pixels_from_baseline = waveform - baseline

    # Calibration: ~17.5 pixels/mm at 300 DPI, 10 mm/mV
    pixels_per_mm = 17.5
    mm = pixels_from_baseline / pixels_per_mm
    voltage = mm / 10.0

    print(f"  ✓ {len(voltage)} samples, range [{voltage.min():.3f}, {voltage.max():.3f}] mV")

    return voltage


def resample_and_concatenate(strips_signals):
    """Resample each strip to 5120 samples and concatenate with baseline alignment"""
    print("\nResampling strips to 512 Hz...")

    resampled = []
    for i, sig in enumerate(strips_signals):
        # Each strip = 10 seconds → 5120 samples at 512 Hz
        duration = 10.0
        target_samples = 5120

        x_old = np.linspace(0, duration, len(sig))
        x_new = np.linspace(0, duration, target_samples)

        f = interpolate.interp1d(x_old, sig, kind='linear', fill_value='extrapolate')
        sig_resampled = f(x_new)

        resampled.append(sig_resampled)
        print(f"  Strip {i+1}: {len(sig)} → {target_samples} samples")

    # Align baselines before concatenation
    print("\nAligning strip baselines...")
    aligned = []

    for i, sig in enumerate(resampled):
        if i == 0:
            # First strip - use as reference
            aligned.append(sig)
            prev_end_mean = np.mean(sig[-100:])  # Mean of last 100 samples
        else:
            # Align this strip to match previous strip's ending
            curr_start_mean = np.mean(sig[:100])  # Mean of first 100 samples
            offset = prev_end_mean - curr_start_mean

            sig_aligned = sig + offset
            aligned.append(sig_aligned)

            print(f"  Strip {i+1}: adjusted by {offset:.4f} mV to match Strip {i}")
            prev_end_mean = np.mean(sig_aligned[-100:])

    # Smooth transitions at boundaries
    print("\nSmoothing strip boundaries...")
    full_signal = aligned[0].copy()

    for i in range(1, len(aligned)):
        next_strip = aligned[i].copy()

        # Apply gentle cross-fade at boundary (50 samples on each side)
        fade_len = 50
        fade = np.linspace(0, 1, fade_len)

        # Blend last 50 samples of previous with first 50 of next
        end_prev = full_signal[-fade_len:]
        start_next = next_strip[:fade_len]

        blended = end_prev * (1 - fade) + start_next * fade

        # Replace boundary region with blend
        full_signal[-fade_len:] = blended

        # Concatenate the rest
        full_signal = np.concatenate([full_signal, next_strip[fade_len:]])

    time = np.arange(len(full_signal)) / 512.0

    print(f"  Total: {len(full_signal)} samples ({time[-1]:.1f}s)")

    return full_signal, time


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


def plot_results(signal, time):
    """Create visualization"""
    fig = plt.figure(figsize=(18, 10))

    # Full signal
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, signal, 'r-', linewidth=0.7)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Voltage (mV)', fontsize=11)
    ax1.set_title(f'Complete ECG Signal - {time[-1]:.1f} seconds', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for t in [10, 20]:
        ax1.axvline(t, color='gray', linestyle='--', alpha=0.5)

    # Three 10s strips
    for i in range(3):
        ax = plt.subplot(3, 3, 4+i)
        mask = (time >= i*10) & (time < (i+1)*10)
        if np.any(mask):
            ax.plot(time[mask], signal[mask], 'r-', linewidth=0.8)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Voltage (mV)', fontsize=9)
            ax.set_title(f'Strip {i+1}: {i*10}-{(i+1)*10}s', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

    # Zoomed
    ax5 = plt.subplot(3, 3, 7)
    mask = time <= 3
    if np.any(mask):
        ax5.plot(time[mask], signal[mask], 'r-', linewidth=1.2)
        ax5.set_xlabel('Time (s)', fontsize=9)
        ax5.set_ylabel('Voltage (mV)', fontsize=9)
        ax5.set_title('First 3 Seconds', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # Histogram
    ax6 = plt.subplot(3, 3, 8)
    ax6.hist(signal, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Voltage (mV)', fontsize=9)
    ax6.set_ylabel('Frequency', fontsize=9)
    ax6.set_title('Distribution', fontsize=10, fontweight='bold')
    ax6.axvline(signal.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {signal.mean():.3f}')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Spectrum
    ax7 = plt.subplot(3, 3, 9)
    freqs = np.fft.rfftfreq(len(signal), 1/512)
    fft = np.abs(np.fft.rfft(signal))
    ax7.semilogy(freqs, fft, linewidth=0.8)
    ax7.set_xlabel('Frequency (Hz)', fontsize=9)
    ax7.set_ylabel('Magnitude', fontsize=9)
    ax7.set_title('Spectrum', fontsize=10, fontweight='bold')
    ax7.set_xlim(0, 50)
    ax7.grid(True, alpha=0.3)

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
    print("FINAL SIMPLE ECG EXTRACTOR")
    print("=" * 70)
    print()

    # 1. Load
    image = load_pdf(pdf_path)

    # 2. Detect waveform pixels
    mask = detect_waveform_pixels(image)

    # 3. Find strips (simple: group consecutive rows with pixels)
    strips = find_strips_simple(mask)

    if len(strips) == 0:
        print("\n❌ ERROR: No strips found!")
        print("Check output/debug/strip_detection_final.png")
        sys.exit(1)

    if len(strips) != 3:
        print(f"\n⚠ WARNING: Found {len(strips)} strips (expected 3)")
        print("Continuing anyway...\n")

    # 4. Extract waveforms
    print("Extracting waveforms...")
    signals = []
    for i, (y_start, y_end) in enumerate(strips):
        sig = extract_waveform(mask, y_start, y_end, i+1)
        signals.append(sig)

    # 5. Resample and concatenate
    full_signal, time = resample_and_concatenate(signals)

    # 6. Filter
    print("\nFiltering...")
    full_signal = apply_filters(full_signal)
    print("  ✓ Filtered")

    # 7. Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nDuration: {time[-1]:.2f} seconds")
    print(f"Samples: {len(full_signal):,}")
    print(f"Voltage: [{full_signal.min():.3f}, {full_signal.max():.3f}] mV")
    print(f"Std: {full_signal.std():.3f} mV")

    hr = estimate_hr(full_signal)
    if hr:
        print(f"Heart Rate: {hr:.1f} bpm")

    # 8. Save
    print("\nSaving...")
    Path("output").mkdir(exist_ok=True)
    np.save('output/extracted_ecg_signal.npy', full_signal)
    np.save('output/extracted_ecg_time.npy', time)
    print("  output/extracted_ecg_signal.npy")
    print("  output/extracted_ecg_time.npy")

    # 9. Plot
    print("\nPlotting...")
    plot_results(full_signal, time)

    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  output/ecg_final.png")
    print("  output/debug/strip_detection_final.png")
    print("  output/debug/mask_strip_*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
