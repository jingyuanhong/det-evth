"""
ECG OCR Module - Convert ECG images to digital signals

This module provides tools to extract ECG waveforms from images
(typically Apple Watch PDF exports) and convert them to digital
time-series signals suitable for AI analysis.
"""

from .ecg_ocr import ECGImageProcessor, ECGMetadata, GridParams

__version__ = "1.0.0"
__all__ = ['ECGImageProcessor', 'ECGMetadata', 'GridParams']
