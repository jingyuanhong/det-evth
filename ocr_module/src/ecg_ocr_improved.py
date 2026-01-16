"""
Improved ECG OCR for Apple Watch PDFs

This version specifically optimizes for Apple Watch ECG format:
- 3 horizontal strips (0-10s, 10-20s, 20-30s)
- Red waveform on pink/gray grid background
- More robust waveform extraction
- Better noise filtering
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def improved_strip_detection(image: np.ndarray, debug=False) -> List[Tuple[int, int]]:
    """
    Improved strip detection using content analysis

    Returns:
        List of (y_start, y_end) tuples for each strip
    """
    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find horizontal projection (sum of pixel intensity per row)
    horizontal_proj = np.sum(gray, axis=1)

    # Normalize
    horizontal_proj = horizontal_proj / width

    # Find regions with content (higher intensity = more content)
    # Use threshold to find ECG strip regions
    threshold = np.percentile(horizontal_proj, 30)  # Bottom 30% are gaps

    # Find continuous regions above threshold
    in_strip = horizontal_proj > threshold

    # Find strip boundaries
    strips = []
    start = None
    min_strip_height = height * 0.15  # At least 15% of image height

    for i in range(len(in_strip)):
        if in_strip[i] and start is None:
            start = i
        elif not in_strip[i] and start is not None:
            if i - start > min_strip_height:
                strips.append((start, i))
            start = None

    # Handle last strip
    if start is not None and len(in_strip) - start > min_strip_height:
        strips.append((start, len(in_strip)))

    logger.info(f"Detected {len(strips)} ECG strips using content analysis")
    for i, (start, end) in enumerate(strips):
        logger.info(f"  Strip {i}: rows {start}-{end} (height: {end-start} px)")

    if debug:
        # Save debug visualization
        debug_img = image.copy()
        for start, end in strips:
            cv2.line(debug_img, (0, start), (width, start), (0, 255, 0), 3)
            cv2.line(debug_img, (0, end), (width, end), (0, 0, 255), 3)
        cv2.imwrite('output/debug_strip_detection.png', debug_img)
        logger.info("Debug image saved: output/debug_strip_detection.png")

    # If we didn't find 3 strips, fall back to simple division
    if len(strips) != 3:
        logger.warning(f"Expected 3 strips, found {len(strips)}. Using fallback method.")
        return fallback_strip_detection(image)

    return strips


def fallback_strip_detection(image: np.ndarray) -> List[Tuple[int, int]]:
    """Fallback: divide into 3 equal strips, skipping header/footer"""
    height = image.shape[0]

    # Skip header (top 12%) and footer (bottom 3%)
    header_offset = int(height * 0.12)
    footer_offset = int(height * 0.03)
    usable_height = height - header_offset - footer_offset

    strip_height = usable_height // 3

    strips = []
    for i in range(3):
        y_start = header_offset + i * strip_height
        y_end = y_start + strip_height
        strips.append((y_start, y_end))

    logger.info("Using fallback strip detection (equal division)")
    return strips


def improved_red_extraction(strip: np.ndarray, debug=False, strip_index=0) -> np.ndarray:
    """
    Improved red waveform extraction with better noise filtering
    """
    height, width = strip.shape[:2]

    # Convert to multiple color spaces for robust detection
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)

    # Method 1: HSV-based red detection (more restrictive)
    # Red hue: 0-10 and 160-180
    lower_red1 = np.array([0, 120, 120])  # Increased saturation threshold
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)

    # Method 2: LAB-based red detection (a channel > threshold)
    # In LAB, red has high 'a' value
    a_channel = lab[:, :, 1]
    lab_mask = (a_channel > 140).astype(np.uint8) * 255

    # Combine both methods
    red_mask = cv2.bitwise_and(hsv_mask, lab_mask)

    # Morphological operations to remove grid lines
    # Grid lines are thin, waveform is thicker
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)

    # Remove small noise
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)

    # Close small gaps in waveform
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_medium)

    # Remove isolated pixels (likely grid intersections)
    red_mask = cv2.medianBlur(red_mask, 5)

    if debug:
        cv2.imwrite(f'output/debug_mask_strip_{strip_index}.png', red_mask)
        logger.info(f"Debug mask saved for strip {strip_index}")

    # Extract waveform by finding dominant red pixel in each column
    waveform = []
    empty_columns = 0

    for x in range(width):
        column = red_mask[:, x]
        red_pixels = np.where(column > 0)[0]

        if len(red_pixels) > 0:
            # Use weighted average of red pixels (weighted by intensity)
            # This helps when there are multiple red pixels in a column
            if len(red_pixels) > 10:
                # Too many red pixels = likely noise, use median
                y_position = np.median(red_pixels)
            else:
                # Use centroid for better accuracy
                y_position = np.mean(red_pixels)
            empty_columns = 0
        else:
            # No red pixel found
            y_position = np.nan
            empty_columns += 1

            # If too many consecutive empty columns, might be in gap between strips
            if empty_columns > 50:
                logger.warning(f"Strip {strip_index}: {empty_columns} consecutive empty columns at x={x}")

        waveform.append(y_position)

    waveform = np.array(waveform)

    # Interpolate missing values
    nans = np.isnan(waveform)
    nan_count = np.sum(nans)

    if nan_count > len(waveform) * 0.3:
        logger.warning(f"Strip {strip_index}: {nan_count}/{len(waveform)} ({nan_count/len(waveform)*100:.1f}%) missing values")

    if nans.any() and not nans.all():
        x_indices = np.arange(len(waveform))
        waveform[nans] = np.interp(x_indices[nans], x_indices[~nans], waveform[~nans])
    elif nans.all():
        logger.error(f"Strip {strip_index}: No valid waveform data found!")
        return np.zeros(width)

    # Invert y-axis (image coordinates to voltage)
    waveform = height - waveform

    # Remove outliers (sudden spikes that are likely noise)
    # Use median absolute deviation
    median = np.median(waveform)
    mad = np.median(np.abs(waveform - median))
    threshold = 5 * mad  # 5 MAD threshold
    outliers = np.abs(waveform - median) > threshold

    if np.any(outliers):
        logger.info(f"Strip {strip_index}: Removing {np.sum(outliers)} outliers")
        waveform[outliers] = median

    logger.info(f"Strip {strip_index}: Extracted waveform with {len(waveform)} samples")

    return waveform


def validate_waveform_quality(waveform: np.ndarray, strip_index: int) -> dict:
    """
    Validate extracted waveform quality

    Returns:
        dict with quality metrics
    """
    # Calculate quality metrics
    std = np.std(waveform)
    mean = np.mean(waveform)

    # A flat line has very low std
    is_flat = std < 1.0

    # Calculate signal variation
    diff = np.diff(waveform)
    variation = np.std(diff)

    # ECG should have periodic variation
    has_variation = variation > 0.5

    quality = {
        'strip_index': strip_index,
        'std': std,
        'mean': mean,
        'variation': variation,
        'is_flat': is_flat,
        'has_variation': has_variation,
        'quality_score': std * variation  # Combined score
    }

    if is_flat:
        logger.warning(f"Strip {strip_index}: Waveform appears FLAT (std={std:.3f})")

    if not has_variation:
        logger.warning(f"Strip {strip_index}: Low variation detected (var={variation:.3f})")

    return quality


# Add these methods to ECGImageProcessor class
def patch_ecg_processor():
    """
    Monkey-patch the ECGImageProcessor class with improved methods
    """
    from ecg_ocr import ECGImageProcessor

    # Replace _detect_strips method
    def new_detect_strips(self) -> List[np.ndarray]:
        """Improved strip detection"""
        logger.info("Using IMPROVED strip detection...")

        if self.image is None:
            raise ValueError("Image not loaded")

        # Get strip boundaries
        strip_bounds = improved_strip_detection(self.image, debug=True)

        # Extract strip images
        strips = []
        for i, (y_start, y_end) in enumerate(strip_bounds):
            strip = self.image[y_start:y_end, :]
            strips.append(strip)
            logger.info(f"Strip {i}: {strip.shape}")

        return strips

    # Replace _extract_red_waveform method
    def new_extract_red_waveform(self, strip: np.ndarray, strip_index: int = 0) -> np.ndarray:
        """Improved red waveform extraction"""
        logger.info(f"Using IMPROVED waveform extraction for strip {strip_index}...")
        waveform = improved_red_extraction(strip, debug=True, strip_index=strip_index)

        # Validate quality
        quality = validate_waveform_quality(waveform, strip_index)
        logger.info(f"Strip {strip_index} quality score: {quality['quality_score']:.2f}")

        return waveform

    # Patch the class
    ECGImageProcessor._detect_strips = new_detect_strips
    ECGImageProcessor._extract_red_waveform = new_extract_red_waveform

    logger.info("ECGImageProcessor patched with improved algorithms")


if __name__ == "__main__":
    print("Improved OCR module loaded")
    print("Import this module and call patch_ecg_processor() to use improved algorithms")
