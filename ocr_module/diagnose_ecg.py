#!/usr/bin/env python3
"""
ECG PDF Diagnostic Tool

This tool analyzes the Apple Watch ECG PDF to understand:
- Image dimensions and layout
- Color distribution
- Where the red waveform actually is
- Grid structure
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, 'src')


def diagnose_ecg_pdf(pdf_path):
    """Comprehensive diagnosis of ECG PDF"""

    print("=" * 70)
    print("ECG PDF DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"\nAnalyzing: {pdf_path}\n")

    # Load PDF
    from pdf2image import convert_from_path
    print("Converting PDF to image...")
    images = convert_from_path(pdf_path, dpi=300)

    if not images:
        print("ERROR: Could not load PDF")
        return

    # Convert to OpenCV format
    pil_image = images[0]
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]
    print(f"✓ Image loaded: {width}x{height} pixels\n")

    # Create output directory
    Path("output/diagnostic").mkdir(parents=True, exist_ok=True)

    # 1. Save original image
    cv2.imwrite('output/diagnostic/01_original.png', image)
    print("1. Saved original image")

    # 2. Analyze color channels
    print("\n2. Color Analysis:")
    analyze_colors(image)

    # 3. Find red pixels
    print("\n3. Red Pixel Detection:")
    red_mask = find_red_pixels(image)
    cv2.imwrite('output/diagnostic/02_red_mask.png', red_mask)
    print("   Saved: output/diagnostic/02_red_mask.png")

    # 4. Horizontal projection (find strips)
    print("\n4. Strip Detection Analysis:")
    analyze_strips(image, red_mask)

    # 5. Visualize grid
    print("\n5. Grid Analysis:")
    analyze_grid(image)

    # 6. Create comprehensive visualization
    print("\n6. Creating diagnostic visualization...")
    create_diagnostic_plot(image, red_mask)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nGenerated files in output/diagnostic/:")
    print("  01_original.png          - Original PDF image")
    print("  02_red_mask.png          - Red color detection mask")
    print("  03_horizontal_proj.png   - Horizontal projection (strip detection)")
    print("  04_grid_lines.png        - Detected grid lines")
    print("  05_diagnostic_full.png   - Complete diagnostic visualization")
    print("\nPlease check these images and share:")
    print("  - 01_original.png (to see actual layout)")
    print("  - 02_red_mask.png (to see if red detection works)")
    print("  - 03_horizontal_proj.png (to see strip locations)")
    print("=" * 70)


def analyze_colors(image):
    """Analyze color distribution in image"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # BGR channels
    b, g, r = cv2.split(image)

    print(f"   Red channel: min={r.min()}, max={r.max()}, mean={r.mean():.1f}")
    print(f"   Green channel: min={g.min()}, max={g.max()}, mean={g.mean():.1f}")
    print(f"   Blue channel: min={b.min()}, max={b.max()}, mean={b.mean():.1f}")

    # HSV
    h, s, v = cv2.split(hsv)
    print(f"   Hue: min={h.min()}, max={h.max()}, mean={h.mean():.1f}")
    print(f"   Saturation: min={s.min()}, max={s.max()}, mean={s.mean():.1f}")
    print(f"   Value: min={v.min()}, max={v.max()}, mean={v.mean():.1f}")

    # Find pixels with high red
    high_red = (r > 150) & (r > g + 30) & (r > b + 30)
    print(f"   Pixels with high red: {np.sum(high_red):,} ({np.sum(high_red)/r.size*100:.2f}%)")


def find_red_pixels(image):
    """Find red pixels using multiple methods"""

    # Method 1: Simple BGR threshold
    b, g, r = cv2.split(image)
    mask1 = ((r > 150) & (r > g + 30) & (r > b + 30)).astype(np.uint8) * 255

    # Method 2: HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2a = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2b = cv2.inRange(hsv, lower_red2, upper_red2)
    mask2 = cv2.bitwise_or(mask2a, mask2b)

    # Combine
    combined = cv2.bitwise_or(mask1, mask2)

    # Show statistics
    print(f"   Method 1 (BGR): {np.sum(mask1 > 0):,} red pixels")
    print(f"   Method 2 (HSV): {np.sum(mask2 > 0):,} red pixels")
    print(f"   Combined: {np.sum(combined > 0):,} red pixels")

    return combined


def analyze_strips(image, red_mask):
    """Analyze horizontal distribution to find strips"""
    height, width = image.shape[:2]

    # Horizontal projection of red pixels
    red_projection = np.sum(red_mask > 0, axis=1)

    # Normalize
    red_projection = red_projection / width

    # Find peaks (likely ECG strips)
    threshold = np.percentile(red_projection, 75)  # Top 25%
    peaks = red_projection > threshold

    # Find continuous regions
    regions = []
    start = None
    for i, val in enumerate(peaks):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(peaks)))

    print(f"   Found {len(regions)} potential ECG strip regions:")
    for i, (start, end) in enumerate(regions):
        print(f"      Region {i}: rows {start}-{end} (height: {end-start} px)")

    # Plot horizontal projection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Left: projection graph
    ax1.plot(red_projection, range(height))
    ax1.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    ax1.set_xlabel('Red Pixel Density')
    ax1.set_ylabel('Row (pixels)')
    ax1.set_title('Horizontal Projection of Red Pixels')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: image with detected regions
    img_with_regions = image.copy()
    for start, end in regions:
        cv2.line(img_with_regions, (0, start), (width, start), (0, 255, 0), 3)
        cv2.line(img_with_regions, (0, end), (width, end), (0, 0, 255), 3)
        # Add text
        cv2.putText(img_with_regions, f'Start', (10, start-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_with_regions, f'End', (10, end+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    img_rgb = cv2.cvtColor(img_with_regions, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_rgb)
    ax2.set_title(f'Detected Strip Regions ({len(regions)} found)')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('output/diagnostic/03_horizontal_proj.png', dpi=150, bbox_inches='tight')
    print("   Saved: output/diagnostic/03_horizontal_proj.png")
    plt.close()


def analyze_grid(image):
    """Analyze grid structure"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect lines
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=50, maxLineGap=10)

    if lines is None:
        print("   No grid lines detected")
        return

    # Draw detected lines
    img_with_lines = image.copy()
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Horizontal
        if abs(y2 - y1) < 5:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
            h_lines.append((y1 + y2) // 2)
        # Vertical
        elif abs(x2 - x1) < 5:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)
            v_lines.append((x1 + x2) // 2)

    print(f"   Detected {len(set(h_lines))} horizontal lines")
    print(f"   Detected {len(set(v_lines))} vertical lines")

    # Calculate spacing
    if len(h_lines) > 1:
        h_spacing = np.median(np.diff(sorted(list(set(h_lines)))))
        print(f"   Average horizontal spacing: {h_spacing:.2f} pixels")

    cv2.imwrite('output/diagnostic/04_grid_lines.png', img_with_lines)
    print("   Saved: output/diagnostic/04_grid_lines.png")


def create_diagnostic_plot(image, red_mask):
    """Create comprehensive diagnostic visualization"""

    fig = plt.figure(figsize=(16, 12))

    # 1. Original image
    ax1 = plt.subplot(3, 2, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Red mask
    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(red_mask, cmap='hot')
    ax2.set_title('Red Pixel Mask', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Red channel
    ax3 = plt.subplot(3, 2, 3)
    b, g, r = cv2.split(image)
    ax3.imshow(r, cmap='Reds')
    ax3.set_title('Red Channel', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Saturation (HSV)
    ax4 = plt.subplot(3, 2, 4)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ax4.imshow(hsv[:,:,1], cmap='jet')
    ax4.set_title('Saturation (HSV)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. Histogram
    ax5 = plt.subplot(3, 2, 5)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax5.plot(hist, color=color, label=color.upper())
    ax5.set_xlabel('Pixel Value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Color Histogram', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Red pixel location scatter
    ax6 = plt.subplot(3, 2, 6)
    red_y, red_x = np.where(red_mask > 0)
    if len(red_x) > 0:
        # Downsample for visualization
        step = max(1, len(red_x) // 5000)
        ax6.scatter(red_x[::step], red_y[::step], c='red', s=0.5, alpha=0.5)
    ax6.set_xlim(0, image.shape[1])
    ax6.set_ylim(image.shape[0], 0)
    ax6.set_xlabel('X coordinate')
    ax6.set_ylabel('Y coordinate')
    ax6.set_title('Red Pixel Locations', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/diagnostic/05_diagnostic_full.png', dpi=150, bbox_inches='tight')
    print("   Saved: output/diagnostic/05_diagnostic_full.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_ecg.py <path_to_ecg.pdf>")
        print("\nExample:")
        print("  python diagnose_ecg.py data/sample_ecg.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    diagnose_ecg_pdf(pdf_path)


if __name__ == "__main__":
    main()
