"""
ECG OCR Module - Convert ECG images/PDFs to digital signals

This module extracts ECG waveforms from images (typically Apple Watch PDF exports)
and converts them to digital time-series signals suitable for the ECGFounder model.

Author: ECG Detection App Team
Date: 2026-01-16
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ECGMetadata:
    """Metadata extracted from ECG image"""
    heart_rate: Optional[float] = None  # bpm
    sampling_rate: Optional[float] = None  # Hz
    amplitude_scale: Optional[float] = None  # mm/mV
    time_scale: Optional[float] = None  # mm/s
    duration: Optional[float] = None  # seconds
    lead_type: Optional[str] = None  # e.g., "Lead-I"
    device_info: Optional[str] = None
    recording_time: Optional[str] = None


@dataclass
class GridParams:
    """ECG grid parameters for calibration"""
    small_square_mm: float = 1.0  # Standard ECG small square size (mm)
    large_square_mm: float = 5.0  # Standard ECG large square size (mm)
    small_square_px: Optional[float] = None  # Detected size in pixels
    large_square_px: Optional[float] = None  # Detected size in pixels
    horizontal_lines: Optional[List[int]] = None  # Y-coordinates of grid lines
    vertical_lines: Optional[List[int]] = None  # X-coordinates of grid lines


class ECGImageProcessor:
    """Main class for processing ECG images and extracting signals"""

    def __init__(self, image_path: str):
        """
        Initialize ECG image processor

        Args:
            image_path: Path to ECG image file (PNG, JPG, PDF)
        """
        self.image_path = image_path
        self.image = None
        self.metadata = ECGMetadata()
        self.grid_params = GridParams()

    def load_image(self) -> np.ndarray:
        """
        Load image from file (handles PDF conversion if needed)

        Returns:
            Image as numpy array (BGR format)
        """
        logger.info(f"Loading image from: {self.image_path}")

        if self.image_path.lower().endswith('.pdf'):
            # Handle PDF files
            return self._load_pdf()
        else:
            # Handle image files
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError(f"Failed to load image: {self.image_path}")
            logger.info(f"Image loaded: {self.image.shape}")
            return self.image

    def _load_pdf(self) -> np.ndarray:
        """
        Convert PDF to image

        Returns:
            First page of PDF as numpy array
        """
        try:
            from pdf2image import convert_from_path
            logger.info("Converting PDF to image...")
            images = convert_from_path(self.image_path, dpi=300)

            if not images:
                raise ValueError("No pages found in PDF")

            # Convert PIL Image to OpenCV format
            pil_image = images[0]
            self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            logger.info(f"PDF converted to image: {self.image.shape}")
            return self.image

        except ImportError:
            raise ImportError("pdf2image not installed. Run: pip install pdf2image")

    def detect_grid(self) -> GridParams:
        """
        Detect ECG grid lines and calculate pixel-to-mm calibration

        Returns:
            GridParams object with detected grid information
        """
        logger.info("Detecting ECG grid...")

        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() first.")

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Detect horizontal lines
        horizontal_lines = self._detect_horizontal_lines(gray)

        # Detect vertical lines
        vertical_lines = self._detect_vertical_lines(gray)

        # Calculate grid spacing in pixels
        if len(horizontal_lines) >= 2:
            h_spacing = np.median(np.diff(sorted(horizontal_lines)))
            self.grid_params.small_square_px = h_spacing
            self.grid_params.large_square_px = h_spacing * 5  # 5 small squares = 1 large square
            logger.info(f"Grid detected - Small square: {h_spacing:.2f}px, Large square: {h_spacing*5:.2f}px")

        self.grid_params.horizontal_lines = horizontal_lines
        self.grid_params.vertical_lines = vertical_lines

        return self.grid_params

    def _detect_horizontal_lines(self, gray: np.ndarray) -> List[int]:
        """
        Detect horizontal grid lines using Hough transform

        Args:
            gray: Grayscale image

        Returns:
            List of y-coordinates of detected lines
        """
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is horizontal (small slope)
                if abs(y2 - y1) < 5:  # Nearly horizontal
                    horizontal_lines.append((y1 + y2) // 2)

        # Remove duplicates and sort
        horizontal_lines = sorted(list(set(horizontal_lines)))
        logger.info(f"Detected {len(horizontal_lines)} horizontal grid lines")

        return horizontal_lines

    def _detect_vertical_lines(self, gray: np.ndarray) -> List[int]:
        """
        Detect vertical grid lines

        Args:
            gray: Grayscale image

        Returns:
            List of x-coordinates of detected lines
        """
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is vertical (small horizontal change)
                if abs(x2 - x1) < 5:  # Nearly vertical
                    vertical_lines.append((x1 + x2) // 2)

        # Remove duplicates and sort
        vertical_lines = sorted(list(set(vertical_lines)))
        logger.info(f"Detected {len(vertical_lines)} vertical grid lines")

        return vertical_lines

    def extract_waveform(self, strip_index: int = 0) -> np.ndarray:
        """
        Extract ECG waveform from a specific strip

        Args:
            strip_index: Index of the strip to extract (0, 1, or 2 for 3-strip layout)

        Returns:
            1D array of voltage values
        """
        logger.info(f"Extracting waveform from strip {strip_index}...")

        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() first.")

        # Detect strips (horizontal regions containing ECG waveforms)
        strips = self._detect_strips()

        if strip_index >= len(strips):
            raise ValueError(f"Strip index {strip_index} out of range (found {len(strips)} strips)")

        strip = strips[strip_index]

        # Extract red waveform from strip
        waveform = self._extract_red_waveform(strip)

        return waveform

    def _detect_strips(self) -> List[np.ndarray]:
        """
        Detect individual ECG strips in the image

        Returns:
            List of image regions (numpy arrays) for each strip
        """
        logger.info("Detecting ECG strips...")

        # For typical 3-strip layout, divide image into 3 horizontal regions
        height = self.image.shape[0]
        width = self.image.shape[1]

        # Estimate strip boundaries (assuming equal spacing)
        # Skip header area (top ~15%) and footer area (bottom ~5%)
        header_offset = int(height * 0.15)
        footer_offset = int(height * 0.05)
        usable_height = height - header_offset - footer_offset

        strip_height = usable_height // 3

        strips = []
        for i in range(3):
            y_start = header_offset + i * strip_height
            y_end = y_start + strip_height
            strip = self.image[y_start:y_end, :]
            strips.append(strip)
            logger.info(f"Strip {i}: y={y_start}-{y_end}")

        return strips

    def _extract_red_waveform(self, strip: np.ndarray) -> np.ndarray:
        """
        Extract red waveform pixels from a strip and convert to 1D signal

        Args:
            strip: Image region containing one ECG strip

        Returns:
            1D array of y-coordinates (inverted to represent voltage)
        """
        logger.info("Extracting red waveform pixels...")

        # Convert BGR to HSV for better color segmentation
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        # Red can appear in two ranges in HSV: 0-10 and 170-180
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Extract waveform by finding the topmost red pixel in each column
        height, width = strip.shape[:2]
        waveform = []

        for x in range(width):
            column = red_mask[:, x]
            red_pixels = np.where(column > 0)[0]

            if len(red_pixels) > 0:
                # Use median of red pixels to be robust to noise
                y_position = np.median(red_pixels)
            else:
                # No red pixel found, interpolate later
                y_position = np.nan

            waveform.append(y_position)

        waveform = np.array(waveform)

        # Interpolate missing values
        nans = np.isnan(waveform)
        if nans.any():
            x_indices = np.arange(len(waveform))
            waveform[nans] = np.interp(x_indices[nans], x_indices[~nans], waveform[~nans])

        # Invert y-axis (image coordinates increase downward, voltage increases upward)
        waveform = height - waveform

        logger.info(f"Waveform extracted: {len(waveform)} samples")

        return waveform

    def pixels_to_voltage(self, waveform_pixels: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to voltage values (mV)

        Args:
            waveform_pixels: Array of y-coordinates in pixels

        Returns:
            Array of voltage values in mV
        """
        if self.grid_params.small_square_px is None:
            raise ValueError("Grid not detected. Call detect_grid() first.")

        if self.metadata.amplitude_scale is None:
            logger.warning("Amplitude scale not set in metadata, using default 10 mm/mV")
            self.metadata.amplitude_scale = 10.0  # Default: 10 mm/mV

        # Calculate voltage
        # 1 small square (1mm) = small_square_px pixels
        # amplitude_scale mm = 1 mV
        # So: voltage (mV) = (pixels / small_square_px) / amplitude_scale

        # First, normalize to baseline (assume middle of waveform is baseline)
        baseline = np.median(waveform_pixels)
        pixels_from_baseline = waveform_pixels - baseline

        # Convert to mm
        mm_from_baseline = pixels_from_baseline / self.grid_params.small_square_px

        # Convert to mV
        voltage_mv = mm_from_baseline / self.metadata.amplitude_scale

        logger.info(f"Converted pixels to voltage: range [{voltage_mv.min():.2f}, {voltage_mv.max():.2f}] mV")

        return voltage_mv

    def pixels_to_time(self, num_pixels: int) -> np.ndarray:
        """
        Generate time array for waveform

        Args:
            num_pixels: Number of pixel samples in waveform

        Returns:
            Array of time values in seconds
        """
        if self.grid_params.small_square_px is None:
            raise ValueError("Grid not detected. Call detect_grid() first.")

        if self.metadata.time_scale is None:
            logger.warning("Time scale not set in metadata, using default 25 mm/s")
            self.metadata.time_scale = 25.0  # Default: 25 mm/s

        # Calculate time
        # time_scale mm = 1 second
        # So: time (s) = (pixels / small_square_px) / time_scale

        pixels_array = np.arange(num_pixels)
        mm_array = pixels_array / self.grid_params.small_square_px
        time_array = mm_array / self.metadata.time_scale

        logger.info(f"Generated time array: 0 to {time_array[-1]:.2f} seconds")

        return time_array

    def resample_signal(self, voltage: np.ndarray, time: np.ndarray, target_fs: float = 512.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample signal to target sampling frequency

        Args:
            voltage: Voltage values in mV
            time: Time values in seconds
            target_fs: Target sampling frequency in Hz

        Returns:
            Tuple of (resampled_voltage, resampled_time)
        """
        from scipy import interpolate

        logger.info(f"Resampling signal to {target_fs} Hz...")

        # Create uniform time array at target sampling rate
        duration = time[-1] - time[0]
        num_samples = int(duration * target_fs)
        time_uniform = np.linspace(time[0], time[-1], num_samples)

        # Interpolate voltage values
        f = interpolate.interp1d(time, voltage, kind='linear', fill_value='extrapolate')
        voltage_uniform = f(time_uniform)

        logger.info(f"Resampled: {len(voltage)} -> {len(voltage_uniform)} samples")

        return voltage_uniform, time_uniform

    def concatenate_strips(self, strip_signals: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate multiple strip signals into continuous signal

        Args:
            strip_signals: List of signal arrays from each strip

        Returns:
            Concatenated signal array
        """
        logger.info(f"Concatenating {len(strip_signals)} strips...")

        # Simply concatenate arrays
        full_signal = np.concatenate(strip_signals)

        logger.info(f"Full signal length: {len(full_signal)} samples")

        return full_signal

    def validate_signal(self, signal: np.ndarray, sampling_rate: float) -> Dict[str, any]:
        """
        Validate extracted signal against metadata

        Args:
            signal: Extracted ECG signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating extracted signal...")

        results = {
            'duration': len(signal) / sampling_rate,
            'estimated_hr': None,
            'metadata_hr': self.metadata.heart_rate,
            'hr_match': False,
            'duration_match': False
        }

        # Calculate duration
        duration = results['duration']
        logger.info(f"Signal duration: {duration:.2f}s")

        # Check duration match (expected ~30s for Apple Watch)
        if self.metadata.duration is not None:
            duration_diff = abs(duration - self.metadata.duration)
            results['duration_match'] = duration_diff < 2.0  # Within 2 seconds
            logger.info(f"Duration match: {results['duration_match']} (expected {self.metadata.duration}s)")

        # Estimate heart rate from signal (count peaks)
        results['estimated_hr'] = self._estimate_heart_rate(signal, sampling_rate)

        # Check heart rate match
        if self.metadata.heart_rate is not None and results['estimated_hr'] is not None:
            hr_diff = abs(results['estimated_hr'] - self.metadata.heart_rate)
            results['hr_match'] = hr_diff < 10  # Within 10 bpm
            logger.info(f"Heart rate match: {results['hr_match']} (estimated {results['estimated_hr']:.1f} bpm, expected {self.metadata.heart_rate} bpm)")

        return results

    def _estimate_heart_rate(self, signal: np.ndarray, sampling_rate: float) -> float:
        """
        Estimate heart rate from signal by counting R-peaks

        Args:
            signal: ECG signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Estimated heart rate in bpm
        """
        from scipy.signal import find_peaks

        # Find peaks (R-waves)
        # Use adaptive threshold based on signal statistics
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        peaks, _ = find_peaks(signal, height=threshold, distance=sampling_rate*0.4)  # Min 0.4s between beats

        if len(peaks) < 2:
            logger.warning("Could not detect enough peaks for heart rate estimation")
            return None

        # Calculate heart rate
        duration = len(signal) / sampling_rate
        num_beats = len(peaks)
        heart_rate = (num_beats / duration) * 60  # beats per minute

        logger.info(f"Detected {num_beats} beats in {duration:.1f}s -> {heart_rate:.1f} bpm")

        return heart_rate

    def process_full_ecg(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete pipeline: load image, extract signal, validate

        Returns:
            Tuple of (signal, time, validation_results)
        """
        logger.info("=" * 50)
        logger.info("Starting full ECG OCR pipeline")
        logger.info("=" * 50)

        # Load image
        self.load_image()

        # Detect grid
        self.detect_grid()

        # Extract waveforms from all 3 strips
        strip_signals_voltage = []

        for i in range(3):
            # Extract pixels
            waveform_pixels = self.extract_waveform(strip_index=i)

            # Convert to voltage
            voltage = self.pixels_to_voltage(waveform_pixels)

            # Generate time array
            time = self.pixels_to_time(len(waveform_pixels))

            # Resample to uniform sampling rate (e.g., 512 Hz for Apple Watch)
            target_fs = self.metadata.sampling_rate or 512.0
            voltage_resampled, time_resampled = self.resample_signal(voltage, time, target_fs)

            strip_signals_voltage.append(voltage_resampled)

        # Concatenate strips
        full_signal = self.concatenate_strips(strip_signals_voltage)

        # Generate full time array
        sampling_rate = self.metadata.sampling_rate or 512.0
        full_time = np.arange(len(full_signal)) / sampling_rate

        # Validate
        validation = self.validate_signal(full_signal, sampling_rate)

        logger.info("=" * 50)
        logger.info("ECG OCR pipeline completed successfully")
        logger.info("=" * 50)

        return full_signal, full_time, validation


def main():
    """Example usage"""
    # Example: Process an ECG image
    processor = ECGImageProcessor("path/to/ecg_image.pdf")

    # Set metadata (can be extracted from image or provided manually)
    processor.metadata.heart_rate = 109  # bpm
    processor.metadata.sampling_rate = 512  # Hz
    processor.metadata.amplitude_scale = 10  # mm/mV
    processor.metadata.time_scale = 25  # mm/s
    processor.metadata.duration = 30  # seconds

    # Process
    signal, time, validation = processor.process_full_ecg()

    # Print results
    print(f"\nExtracted ECG Signal:")
    print(f"  Samples: {len(signal)}")
    print(f"  Duration: {validation['duration']:.2f}s")
    print(f"  Heart Rate: {validation['estimated_hr']:.1f} bpm (expected: {validation['metadata_hr']} bpm)")
    print(f"  Validation: HR match={validation['hr_match']}, Duration match={validation['duration_match']}")

    # Save signal
    np.save("extracted_ecg_signal.npy", signal)
    print("\nSignal saved to: extracted_ecg_signal.npy")


if __name__ == "__main__":
    main()
