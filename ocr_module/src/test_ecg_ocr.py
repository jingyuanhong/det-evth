"""
Test script for ECG OCR module

Usage:
    python test_ecg_ocr.py --image path/to/ecg.pdf
    python test_ecg_ocr.py --image path/to/ecg.pdf --visualize
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Import ECG OCR module
from ecg_ocr import ECGImageProcessor, ECGMetadata


def test_ecg_ocr(image_path: str, visualize: bool = False, save_output: bool = True):
    """
    Test ECG OCR pipeline with a sample image

    Args:
        image_path: Path to ECG image/PDF file
        visualize: Whether to display visualization
        save_output: Whether to save extracted signal to file
    """
    print("=" * 70)
    print("ECG OCR Test Script")
    print("=" * 70)
    print(f"\nInput file: {image_path}\n")

    # Initialize processor
    processor = ECGImageProcessor(image_path)

    # Set metadata (Apple Watch ECG standard values)
    # These can be extracted from the image or provided manually
    processor.metadata.heart_rate = 109  # bpm (from sample)
    processor.metadata.sampling_rate = 512  # Hz (Apple Watch standard)
    processor.metadata.amplitude_scale = 10  # mm/mV (standard)
    processor.metadata.time_scale = 25  # mm/s (standard)
    processor.metadata.duration = 30  # seconds (typical Apple Watch)

    print("Metadata:")
    print(f"  Heart Rate: {processor.metadata.heart_rate} bpm")
    print(f"  Sampling Rate: {processor.metadata.sampling_rate} Hz")
    print(f"  Amplitude Scale: {processor.metadata.amplitude_scale} mm/mV")
    print(f"  Time Scale: {processor.metadata.time_scale} mm/s")
    print(f"  Duration: {processor.metadata.duration} s")
    print()

    # Run full pipeline
    try:
        signal, time, validation = processor.process_full_ecg()

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nExtracted Signal:")
        print(f"  Total Samples: {len(signal)}")
        print(f"  Duration: {validation['duration']:.2f} seconds")
        print(f"  Sampling Rate: {len(signal) / validation['duration']:.1f} Hz")
        print(f"  Voltage Range: [{signal.min():.3f}, {signal.max():.3f}] mV")
        print(f"  Mean: {signal.mean():.3f} mV")
        print(f"  Std Dev: {signal.std():.3f} mV")

        print(f"\nValidation:")
        print(f"  Estimated Heart Rate: {validation['estimated_hr']:.1f} bpm")
        print(f"  Expected Heart Rate: {validation['metadata_hr']} bpm")
        print(f"  Heart Rate Match: {'✓ PASS' if validation['hr_match'] else '✗ FAIL'}")
        print(f"  Duration Match: {'✓ PASS' if validation['duration_match'] else '✗ FAIL'}")

        overall_pass = validation['hr_match'] and validation['duration_match']
        print(f"\n  Overall: {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}")

        # Save output
        if save_output:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            signal_path = output_dir / "extracted_ecg_signal.npy"
            time_path = output_dir / "extracted_ecg_time.npy"

            np.save(signal_path, signal)
            np.save(time_path, time)

            print(f"\nOutput saved:")
            print(f"  Signal: {signal_path}")
            print(f"  Time: {time_path}")

        # Visualize
        if visualize:
            visualize_results(signal, time, validation, processor)

        return signal, time, validation

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def visualize_results(signal, time, validation, processor):
    """
    Create visualization of extracted ECG signal

    Args:
        signal: Extracted ECG signal
        time: Time array
        validation: Validation results
        processor: ECGImageProcessor instance
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        print("\nGenerating visualization...")

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Original image
        ax1 = fig.add_subplot(gs[0, :])
        if processor.image is not None:
            import cv2
            img_rgb = cv2.cvtColor(processor.image, cv2.COLOR_BGR2RGB)
            ax1.imshow(img_rgb)
            ax1.set_title("Original ECG Image", fontsize=14, fontweight='bold')
            ax1.axis('off')

        # 2. Full extracted signal
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(time, signal, 'r-', linewidth=0.8)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Voltage (mV)', fontsize=12)
        ax2.set_title(f'Extracted ECG Signal - {validation["duration"]:.1f}s, HR: {validation["estimated_hr"]:.1f} bpm',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Zoomed view (first 5 seconds)
        ax3 = fig.add_subplot(gs[2, 0])
        mask = time <= 5.0
        ax3.plot(time[mask], signal[mask], 'r-', linewidth=1.2)
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Voltage (mV)', fontsize=12)
        ax3.set_title('Zoomed View (0-5 seconds)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Signal statistics
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(signal, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Voltage (mV)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Signal Distribution', fontsize=12, fontweight='bold')
        ax4.axvline(signal.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.legend()

        plt.tight_layout()

        # Save figure
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        fig_path = output_dir / "ecg_ocr_visualization.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved: {fig_path}")

        plt.show()

    except ImportError:
        print("  Matplotlib not installed. Install with: pip install matplotlib")
    except Exception as e:
        print(f"  Visualization error: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test ECG OCR module')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to ECG image/PDF file')
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization of results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output files')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.image).exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    # Run test
    signal, time, validation = test_ecg_ocr(
        args.image,
        visualize=args.visualize,
        save_output=not args.no_save
    )

    # Exit code based on validation
    if validation is not None:
        success = validation.get('hr_match', False) and validation.get('duration_match', False)
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
