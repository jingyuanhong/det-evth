#!/usr/bin/env python3
"""
Optimized ECG extractor specifically for Apple Watch PDF format

Based on diagnostic analysis, this extractor is tailored for the specific
layout observed in Apple Watch ECG PDFs.
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate, signal as scipy_signal
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AppleWatchECGExtractor:
    """Optimized extractor for Apple Watch ECG PDFs"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.image = None
        self.strips = []
        self.signals = []

    def load_pdf(self):
        """Load PDF and convert to image"""
        from pdf2image import convert_from_path

        logger.info(f"Loading PDF: {self.pdf_path}")
        images = convert_from_path(self.pdf_path, dpi=300)

        if not images:
            raise ValueError("Could not load PDF")

        # Convert to OpenCV format
        pil_image = images[0]
        self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        height, width = self.image.shape[:2]
        logger.info(f"✓ Image loaded: {width}x{height} pixels")

    def detect_strips_fixed(self):
        """
        Detect ECG strips using FIXED positions based on diagnostic analysis

        From the horizontal projection, we know the strips are at:
        - Strip 1: ~rows 300-600 (approximate, needs to be detected)
        - Strip 2: ~rows 1000-1700
        - Strip 3: ~rows 2000-2300

        But let's use a more robust approach: find the actual content regions.
        """
        height, width = self.image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Calculate horizontal projection (sum of all pixels per row)
        # Lower values = more content (darker)
        horizontal_proj = np.mean(gray, axis=1)

        # Find rows with significant content (darker than threshold)
        # Most of image is white background (~240-250), ECG strips are darker
        content_threshold = np.percentile(horizontal_proj, 40)  # Bottom 40% brightness
        has_content = horizontal_proj < content_threshold

        # Find continuous content regions
        regions = []
        in_region = False
        start = None
        min_height = height * 0.10  # At least 10% of image height

        for i in range(len(has_content)):
            if has_content[i] and not in_region:
                start = i
                in_region = True
            elif not has_content[i] and in_region:
                if i - start > min_height:
                    regions.append((start, i))
                in_region = False
                start = None

        # Handle last region
        if in_region and len(has_content) - start > min_height:
            regions.append((start, len(has_content)))

        logger.info(f"Detected {len(regions)} content regions:")
        for i, (start, end) in enumerate(regions):
            logger.info(f"  Region {i+1}: rows {start}-{end} (height: {end-start} px)")

        # Extract strip images
        self.strips = []
        for i, (y_start, y_end) in enumerate(regions):
            # Add small padding
            padding = 20
            y_start = max(0, y_start - padding)
            y_end = min(height, y_end + padding)

            strip = self.image[y_start:y_end, :]
            self.strips.append({
                'index': i,
                'image': strip,
                'y_start': y_start,
                'y_end': y_end
            })

        return len(self.strips)

    def extract_red_waveform_precise(self, strip_data):
        """Extract red waveform with precise color detection"""
        strip = strip_data['image']
        height, width = strip.shape[:2]
        strip_index = strip_data['index']

        logger.info(f"Extracting waveform from strip {strip_index+1}...")

        # Convert to HSV and LAB
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)

        # Method 1: HSV red detection (strict)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        mask_hsv1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_hsv2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # Method 2: LAB a-channel (red)
        a_channel = lab[:, :, 1]
        mask_lab = (a_channel > 135).astype(np.uint8) * 255

        # Method 3: Simple BGR (red dominant)
        b, g, r = cv2.split(strip)
        mask_bgr = ((r > 130) & (r > g + 20) & (r > b + 20)).astype(np.uint8) * 255

        # Combine all methods (at least 2 out of 3 agree)
        combined = ((mask_hsv > 0).astype(int) +
                   (mask_lab > 0).astype(int) +
                   (mask_bgr > 0).astype(int))
        red_mask = (combined >= 2).astype(np.uint8) * 255

        # Remove noise with morphology
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_small)

        # Median blur to remove isolated pixels
        red_mask = cv2.medianBlur(red_mask, 3)

        # Save debug mask
        Path("output/debug").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'output/debug/mask_strip_{strip_index+1}.png', red_mask)

        # Extract waveform column by column
        waveform = []
        for x in range(width):
            column = red_mask[:, x]
            red_pixels = np.where(column > 0)[0]

            if len(red_pixels) > 0:
                # Use median for robustness
                if len(red_pixels) > 8:
                    # Too many pixels = noise, use median
                    y_pos = np.median(red_pixels)
                else:
                    # Use weighted average
                    weights = column[red_pixels].astype(float)
                    y_pos = np.average(red_pixels, weights=weights)
            else:
                y_pos = np.nan

            waveform.append(y_pos)

        waveform = np.array(waveform)

        # Interpolate missing values
        nans = np.isnan(waveform)
        if nans.any() and not nans.all():
            x_idx = np.arange(len(waveform))
            waveform[nans] = np.interp(x_idx[nans], x_idx[~nans], waveform[~nans])

        # Invert y-axis (image coords to voltage)
        waveform = height - waveform

        # Remove outliers using MAD
        median = np.median(waveform)
        mad = np.median(np.abs(waveform - median))
        if mad > 0:
            outliers = np.abs(waveform - median) > (5 * mad)
            if np.any(outliers):
                logger.info(f"  Removed {np.sum(outliers)} outliers")
                waveform[outliers] = median

        logger.info(f"  ✓ Extracted {len(waveform)} samples")

        return waveform

    def pixels_to_voltage(self, waveform_pixels, strip_height):
        """Convert pixels to voltage (mV)"""
        # Detect grid spacing from the strip
        # For now, use metadata: 10 mm/mV, need to find pixels per mm

        # Normalize to baseline (median)
        baseline = np.median(waveform_pixels)
        pixels_from_baseline = waveform_pixels - baseline

        # From metadata: 10 mm/mV, 25 mm/s
        # Need to estimate pixels per mm from grid
        # Typical: 1 small square = 1mm, appears as ~10-20 pixels at 300 DPI

        # For Apple Watch: empirically, grid spacing is about 15-20 pixels per mm
        pixels_per_mm = 17  # Estimated from 300 DPI and typical grid

        # Convert to mm
        mm_from_baseline = pixels_from_baseline / pixels_per_mm

        # Convert to mV (10 mm = 1 mV)
        voltage_mv = mm_from_baseline / 10.0

        return voltage_mv

    def resample_to_512hz(self, signal_list):
        """
        Resample signals to 512 Hz and concatenate

        Each strip represents 10 seconds at 25 mm/s
        At 300 DPI: ~2953 pixels width per strip (25mm * 10s * 300/25.4)
        So pixel rate ≈ 295 pixels/second
        Need to resample to 512 samples/second for 10 seconds = 5120 samples
        """
        resampled = []

        for i, (signal, strip_info) in enumerate(signal_list):
            width = len(signal)

            # Each strip is 10 seconds
            duration = 10.0  # seconds
            target_samples = int(duration * 512)  # 5120 samples

            # Resample using linear interpolation
            x_old = np.linspace(0, duration, len(signal))
            x_new = np.linspace(0, duration, target_samples)

            f = interpolate.interp1d(x_old, signal, kind='linear', fill_value='extrapolate')
            signal_resampled = f(x_new)

            resampled.append(signal_resampled)
            logger.info(f"Strip {i+1}: {len(signal)} → {len(signal_resampled)} samples")

        # Concatenate all strips
        full_signal = np.concatenate(resampled)
        logger.info(f"✓ Total signal: {len(full_signal)} samples ({len(full_signal)/512:.1f} seconds)")

        return full_signal

    def apply_filters(self, signal, fs=512):
        """Apply basic filtering to clean up signal"""
        # High-pass filter to remove baseline drift (0.5 Hz)
        sos_high = scipy_signal.butter(2, 0.5, 'highpass', fs=fs, output='sos')
        signal_filtered = scipy_signal.sosfilt(sos_high, signal)

        # Low-pass filter to remove high-frequency noise (40 Hz)
        sos_low = scipy_signal.butter(2, 40, 'lowpass', fs=fs, output='sos')
        signal_filtered = scipy_signal.sosfilt(sos_low, signal_filtered)

        return signal_filtered

    def extract_full_ecg(self):
        """Complete extraction pipeline"""
        logger.info("=" * 70)
        logger.info("APPLE WATCH ECG EXTRACTION")
        logger.info("=" * 70)

        # 1. Load PDF
        self.load_pdf()

        # 2. Detect strips
        logger.info("\nDetecting ECG strips...")
        num_strips = self.detect_strips_fixed()

        if num_strips != 3:
            logger.warning(f"⚠ Expected 3 strips, found {num_strips}")

        # 3. Extract waveforms
        logger.info("\nExtracting waveforms...")
        signal_data = []

        for strip_data in self.strips:
            waveform_pixels = self.extract_red_waveform_precise(strip_data)

            # Convert to voltage
            voltage = self.pixels_to_voltage(waveform_pixels, strip_data['image'].shape[0])

            signal_data.append((voltage, strip_data))

        # 4. Resample and concatenate
        logger.info("\nResampling to 512 Hz...")
        full_signal = self.resample_to_512hz(signal_data)

        # 5. Apply filters
        logger.info("\nApplying filters...")
        full_signal_filtered = self.apply_filters(full_signal)

        # 6. Generate time array
        time = np.arange(len(full_signal_filtered)) / 512.0

        logger.info("\n" + "=" * 70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nSignal: {len(full_signal_filtered)} samples")
        logger.info(f"Duration: {time[-1]:.2f} seconds")
        logger.info(f"Voltage range: [{full_signal_filtered.min():.3f}, {full_signal_filtered.max():.3f}] mV")
        logger.info(f"Std dev: {full_signal_filtered.std():.3f} mV")

        return full_signal_filtered, time

    def save_results(self, signal, time):
        """Save extracted signal"""
        Path("output").mkdir(exist_ok=True)

        np.save('output/extracted_ecg_signal.npy', signal)
        np.save('output/extracted_ecg_time.npy', time)

        logger.info("\nSaved:")
        logger.info("  output/extracted_ecg_signal.npy")
        logger.info("  output/extracted_ecg_time.npy")

    def plot_results(self, signal, time):
        """Create visualization"""
        fig = plt.figure(figsize=(16, 10))

        # Full signal
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(time, signal, 'r-', linewidth=0.8)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_title(f'Complete ECG Signal ({time[-1]:.1f} seconds)', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Three 10-second segments
        for i in range(3):
            ax = plt.subplot(4, 3, i+4)
            mask = (time >= i*10) & (time < (i+1)*10)
            ax.plot(time[mask], signal[mask], 'r-', linewidth=1.0)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage (mV)')
            ax.set_title(f'Strip {i+1} ({i*10}-{(i+1)*10}s)', fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Zoomed view (first 3 seconds)
        ax5 = plt.subplot(4, 3, 7)
        mask = time <= 3
        ax5.plot(time[mask], signal[mask], 'r-', linewidth=1.2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Voltage (mV)')
        ax5.set_title('First 3 seconds (detailed)', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Histogram
        ax6 = plt.subplot(4, 3, 8)
        ax6.hist(signal, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Voltage (mV)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Voltage Distribution', fontweight='bold')
        ax6.axvline(signal.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {signal.mean():.3f}')
        ax6.legend()

        # Power spectrum
        ax7 = plt.subplot(4, 3, 9)
        freqs = np.fft.rfftfreq(len(signal), 1/512)
        fft = np.abs(np.fft.rfft(signal))
        ax7.plot(freqs, fft)
        ax7.set_xlabel('Frequency (Hz)')
        ax7.set_ylabel('Magnitude')
        ax7.set_title('Frequency Spectrum', fontweight='bold')
        ax7.set_xlim(0, 50)
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('output/apple_watch_ecg_final.png', dpi=150, bbox_inches='tight')
        logger.info("  output/apple_watch_ecg_final.png")
        plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_apple_watch_ecg.py <path_to_ecg.pdf>")
        print("\nExample:")
        print("  python extract_apple_watch_ecg.py data/sample_ecg.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Extract ECG
    extractor = AppleWatchECGExtractor(pdf_path)
    signal, time = extractor.extract_full_ecg()

    # Save and plot
    extractor.save_results(signal, time)
    extractor.plot_results(signal, time)

    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print("\nCheck output/ folder for results:")
    print("  - extracted_ecg_signal.npy")
    print("  - extracted_ecg_time.npy")
    print("  - apple_watch_ecg_final.png")
    print("  - debug/mask_strip_*.png")


if __name__ == "__main__":
    main()
