# ECG OCR Quick Start Guide

Get started with ECG image-to-signal conversion in 5 minutes.

## Installation

```bash
# 1. Navigate to the ocr_module directory
cd ocr_module

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install poppler (for PDF support)
# macOS:
brew install poppler

# Ubuntu/Debian:
sudo apt-get install poppler-utils
```

## Basic Usage

### Test with Your ECG Image

```bash
# Run test script with your Apple Watch ECG PDF
python src/test_ecg_ocr.py --image /path/to/your_ecg.pdf --visualize

# Example with the sample:
python src/test_ecg_ocr.py --image ../data/sample_ecg.pdf --visualize
```

### Python API

```python
from src.ecg_ocr import ECGImageProcessor

# 1. Load your ECG image
processor = ECGImageProcessor("your_ecg.pdf")

# 2. Set metadata (from Apple Watch: typically 512Hz, 25mm/s, 10mm/mV)
processor.metadata.sampling_rate = 512  # Hz
processor.metadata.time_scale = 25      # mm/s
processor.metadata.amplitude_scale = 10  # mm/mV
processor.metadata.heart_rate = 109      # bpm (optional, for validation)
processor.metadata.duration = 30         # seconds (optional)

# 3. Process ECG (one line!)
signal, time, validation = processor.process_full_ecg()

# 4. Use the signal
print(f"Extracted {len(signal)} samples at {512} Hz")
print(f"Duration: {time[-1]:.1f} seconds")
print(f"Heart rate: {validation['estimated_hr']:.1f} bpm")

# 5. Save for later use
import numpy as np
np.save("my_ecg_signal.npy", signal)
```

## What You Get

### Output Files (in `output/` directory)

1. **extracted_ecg_signal.npy** - ECG signal array (mV)
   - Shape: (N,) where N ≈ 15,360 for 30s at 512 Hz
   - Type: float64 numpy array
   - Units: millivolts

2. **extracted_ecg_time.npy** - Time array (seconds)
   - Shape: (N,)
   - Type: float64 numpy array
   - Range: 0 to ~30 seconds

3. **ecg_ocr_visualization.png** - Visual validation
   - Original image
   - Extracted signal plot
   - Zoomed view
   - Signal statistics

### Validation Results

The pipeline automatically validates:
- ✓ Heart rate matches metadata (±10 bpm)
- ✓ Duration matches expected (±2 seconds)
- ✓ Signal quality indicators

## Next Steps

### For ECGFounder Integration

The extracted signal needs preprocessing before feeding to ECGFounder:

```python
# After OCR extraction:
signal, time, _ = processor.process_full_ecg()

# Apply ECGFounder preprocessing (see PROJECT_SPECIFICATION.md):
# 1. Resample 512 Hz → 500 Hz
# 2. High-pass filter (0.5 Hz)
# 3. Low-pass filter (50 Hz Butterworth)
# 4. Notch filter (50/60 Hz)
# 5. Segment into 10-second windows
# 6. Z-score normalize

# Then feed to model
```

See `dataset.py` from ECGFounder repository for full preprocessing code.

### Troubleshooting

**No output or errors?**
- Check that your image file exists and is readable
- Ensure poppler is installed for PDF support
- Verify Python 3.8+ is being used

**Heart rate mismatch?**
- This is normal for initial testing
- Adjust peak detection threshold in `_estimate_heart_rate()`
- Manually verify the signal looks correct visually

**Poor signal quality?**
- Check that input image has high resolution (300 DPI recommended)
- Ensure red waveform is clearly visible
- Try adjusting HSV color ranges in `_extract_red_waveform()`

## Example: Complete Workflow

```python
from src.ecg_ocr import ECGImageProcessor
import numpy as np

# Process ECG
processor = ECGImageProcessor("apple_watch_ecg.pdf")
processor.metadata.sampling_rate = 512
processor.metadata.time_scale = 25
processor.metadata.amplitude_scale = 10

signal, time, validation = processor.process_full_ecg()

# Check validation
if validation['hr_match'] and validation['duration_match']:
    print("✓ Extraction successful!")

    # Save for ECGFounder
    np.save("ecg_for_model.npy", signal)

    # Continue with preprocessing...
else:
    print("⚠ Validation warnings - check signal quality")
    # Still usable, but may need manual review
```

## Resources

- Full documentation: See `README.md`
- Project specification: See `../PROJECT_SPECIFICATION.md`
- ECGFounder model: https://github.com/PKUDigitalHealth/ECGFounder

## Support

For issues or questions, see the main project README.
