#!/usr/bin/env python3
"""
Simple demo to test ECG OCR module

This script demonstrates the OCR functionality.
To test with your actual ECG file:
  python demo_test.py /path/to/your/ecg.pdf
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ecg_ocr import ECGImageProcessor


def demo_ocr(image_path=None):
    """
    Demonstrate ECG OCR functionality

    Args:
        image_path: Path to ECG image/PDF file
    """

    if image_path is None:
        print("=" * 70)
        print("ECG OCR Module - Demo")
        print("=" * 70)
        print("\nNo image file provided.")
        print("\nTo test the OCR module with your ECG file:")
        print("\n  1. Save your ECG PDF to: ocr_module/data/sample_ecg.pdf")
        print("  2. Run: python demo_test.py data/sample_ecg.pdf")
        print("\nOr run directly:")
        print("  python demo_test.py /path/to/your/ecg.pdf")
        print("\n" + "=" * 70)
        return

    # Check if file exists
    if not Path(image_path).exists():
        print(f"ERROR: File not found: {image_path}")
        print("\nPlease check the file path and try again.")
        return

    print("=" * 70)
    print("ECG OCR Module - Processing ECG Image")
    print("=" * 70)
    print(f"\nInput file: {image_path}")
    print()

    # Initialize processor
    print("Step 1: Initializing ECG processor...")
    processor = ECGImageProcessor(image_path)

    # Set metadata (Apple Watch standard values)
    print("Step 2: Setting metadata...")
    processor.metadata.heart_rate = 109  # bpm (from your sample)
    processor.metadata.sampling_rate = 512  # Hz (Apple Watch standard)
    processor.metadata.amplitude_scale = 10  # mm/mV (standard ECG)
    processor.metadata.time_scale = 25  # mm/s (standard ECG)
    processor.metadata.duration = 30  # seconds

    print("  ✓ Heart Rate: {} bpm".format(processor.metadata.heart_rate))
    print("  ✓ Sampling Rate: {} Hz".format(processor.metadata.sampling_rate))
    print("  ✓ Amplitude Scale: {} mm/mV".format(processor.metadata.amplitude_scale))
    print("  ✓ Time Scale: {} mm/s".format(processor.metadata.time_scale))
    print()

    # Process ECG
    try:
        print("Step 3: Processing ECG image...")
        print("  - Loading image...")
        print("  - Detecting grid...")
        print("  - Extracting waveforms (3 strips)...")
        print("  - Converting to voltage...")
        print("  - Resampling to {} Hz...".format(processor.metadata.sampling_rate))
        print("  - Validating signal...")
        print()

        signal, time, validation = processor.process_full_ecg()

        print("=" * 70)
        print("SUCCESS! ECG Extracted")
        print("=" * 70)
        print()
        print("Signal Information:")
        print("  Total Samples: {:,}".format(len(signal)))
        print("  Duration: {:.2f} seconds".format(validation['duration']))
        print("  Actual Sampling Rate: {:.1f} Hz".format(len(signal) / validation['duration']))
        print("  Voltage Range: [{:.3f}, {:.3f}] mV".format(signal.min(), signal.max()))
        print("  Mean Voltage: {:.3f} mV".format(signal.mean()))
        print("  Std Deviation: {:.3f} mV".format(signal.std()))
        print()

        print("Validation Results:")
        print("  Estimated Heart Rate: {:.1f} bpm".format(validation['estimated_hr'] or 0))
        print("  Expected Heart Rate: {} bpm".format(validation['metadata_hr']))

        if validation['hr_match']:
            print("  Heart Rate Match: ✓ PASS (within ±10 bpm)")
        else:
            print("  Heart Rate Match: ⚠ WARNING (differs by >{} bpm)".format(
                abs(validation['estimated_hr'] - validation['metadata_hr']) if validation['estimated_hr'] else 'N/A'
            ))

        if validation['duration_match']:
            print("  Duration Match: ✓ PASS (within ±2s)")
        else:
            print("  Duration Match: ⚠ WARNING")

        print()

        # Save output
        import numpy as np
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        signal_file = output_dir / "extracted_ecg_signal.npy"
        time_file = output_dir / "extracted_ecg_time.npy"

        np.save(signal_file, signal)
        np.save(time_file, time)

        print("Output Files Saved:")
        print("  Signal: {}".format(signal_file))
        print("  Time: {}".format(time_file))
        print()

        # Show how to load
        print("To load the signal later:")
        print("  import numpy as np")
        print("  signal = np.load('{}')".format(signal_file))
        print("  time = np.load('{}')".format(time_file))
        print()

        print("=" * 70)
        print("Next Steps:")
        print("  1. Visualize: python src/test_ecg_ocr.py --image {} --visualize".format(image_path))
        print("  2. Integrate with ECGFounder preprocessing pipeline")
        print("  3. Feed to AI model for disease detection")
        print("=" * 70)

        return signal, time, validation

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Processing failed")
        print("=" * 70)
        print(f"\nError: {str(e)}")
        print("\nCommon issues:")
        print("  - Make sure the file is a valid ECG image/PDF")
        print("  - Check that the image has a red ECG waveform on grid")
        print("  - Ensure poppler-utils is installed for PDF support")
        print()
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = None

    demo_ocr(image_path)
