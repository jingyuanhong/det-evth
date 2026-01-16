#!/usr/bin/env python3
"""
Test improved OCR algorithm on Apple Watch ECG PDF

This uses the enhanced algorithms with:
- Better strip detection
- More accurate red waveform extraction
- Noise filtering
- Quality validation
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ecg_ocr import ECGImageProcessor
from ecg_ocr_improved import patch_ecg_processor

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_improved_ocr(image_path: str):
    """Test improved OCR on ECG image"""

    print("=" * 70)
    print("IMPROVED ECG OCR Test")
    print("=" * 70)
    print(f"\nInput: {image_path}\n")

    # Patch the processor with improved algorithms
    print("Loading improved OCR algorithms...")
    patch_ecg_processor()
    print("✓ Improved algorithms loaded\n")

    # Create processor
    processor = ECGImageProcessor(image_path)

    # Set metadata
    processor.metadata.heart_rate = 109
    processor.metadata.sampling_rate = 512
    processor.metadata.amplitude_scale = 10
    processor.metadata.time_scale = 25
    processor.metadata.duration = 30

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    try:
        # Process ECG
        print("Processing ECG image...")
        print("-" * 70)

        signal, time, validation = processor.process_full_ecg()

        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)

        # Print results
        print(f"\nSignal Properties:")
        print(f"  Samples: {len(signal):,}")
        print(f"  Duration: {time[-1]:.2f} seconds")
        print(f"  Sampling Rate: {len(signal)/time[-1]:.1f} Hz")
        print(f"  Voltage Range: [{signal.min():.3f}, {signal.max():.3f}] mV")
        print(f"  Mean: {signal.mean():.3f} mV")
        print(f"  Std Dev: {signal.std():.3f} mV")

        print(f"\nValidation:")
        print(f"  Estimated HR: {validation['estimated_hr']:.1f} bpm" if validation['estimated_hr'] else "  Estimated HR: N/A")
        print(f"  Expected HR: {validation['metadata_hr']} bpm")
        print(f"  Duration: {validation['duration']:.1f}s (expected: {processor.metadata.duration}s)")

        # Quality check
        print(f"\nQuality Checks:")
        duration_ok = abs(validation['duration'] - 30) < 5  # Within 5 seconds
        voltage_ok = 0.1 < signal.std() < 5.0  # Reasonable voltage variation
        not_flat = signal.std() > 0.01  # Not a flat line

        print(f"  Duration check: {'✓ PASS' if duration_ok else '✗ FAIL'}")
        print(f"  Voltage variation: {'✓ PASS' if voltage_ok else '✗ FAIL'}")
        print(f"  Signal not flat: {'✓ PASS' if not_flat else '✗ FAIL'}")

        all_ok = duration_ok and voltage_ok and not_flat
        print(f"\n  Overall: {'✓ ALL QUALITY CHECKS PASSED' if all_ok else '⚠ SOME CHECKS FAILED'}")

        # Save signal
        np.save('output/extracted_ecg_signal.npy', signal)
        np.save('output/extracted_ecg_time.npy', time)
        print(f"\nSaved:")
        print(f"  output/extracted_ecg_signal.npy")
        print(f"  output/extracted_ecg_time.npy")

        # Create improved visualization
        create_detailed_plot(signal, time, validation)

        print("\n" + "=" * 70)
        print("Next steps:")
        print("  1. Check output/ecg_improved_plot.png")
        print("  2. Check output/debug_*.png for diagnostic images")
        print("  3. If quality is good, proceed to ECGFounder preprocessing")
        print("=" * 70)

        return signal, time, validation

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_detailed_plot(signal, time, validation):
    """Create detailed visualization with multiple views"""

    fig = plt.figure(figsize=(16, 10))

    # 1. Full signal
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, signal, 'r-', linewidth=0.8)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Voltage (mV)', fontsize=11)
    ax1.set_title(f'Complete ECG Signal ({time[-1]:.1f} seconds)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. First 5 seconds (zoomed)
    ax2 = plt.subplot(3, 2, 3)
    mask1 = time <= 5.0
    if np.any(mask1):
        ax2.plot(time[mask1], signal[mask1], 'r-', linewidth=1.2)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Voltage (mV)', fontsize=10)
        ax2.set_title('First 5 seconds', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # 3. Middle 5 seconds (12-17s)
    ax3 = plt.subplot(3, 2, 4)
    mask2 = (time >= 12) & (time <= 17)
    if np.any(mask2):
        ax3.plot(time[mask2], signal[mask2], 'r-', linewidth=1.2)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Voltage (mV)', fontsize=10)
        ax3.set_title('Middle 5 seconds (12-17s)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Last 5 seconds
    ax4 = plt.subplot(3, 2, 5)
    mask3 = time >= (time[-1] - 5)
    if np.any(mask3):
        ax4.plot(time[mask3], signal[mask3], 'r-', linewidth=1.2)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Voltage (mV)', fontsize=10)
        ax4.set_title(f'Last 5 seconds ({time[-1]-5:.1f}-{time[-1]:.1f}s)', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # 5. Signal histogram
    ax5 = plt.subplot(3, 2, 6)
    ax5.hist(signal, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Voltage (mV)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Voltage Distribution', fontsize=11, fontweight='bold')
    ax5.axvline(signal.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {signal.mean():.3f}')
    ax5.legend(fontsize=9)

    plt.tight_layout()

    # Save
    output_path = 'output/ecg_improved_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nDetailed plot saved: {output_path}")

    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_improved_ocr.py <path_to_ecg.pdf>")
        print("\nExample:")
        print("  python test_improved_ocr.py data/sample_ecg.pdf")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    test_improved_ocr(image_path)


if __name__ == "__main__":
    main()
