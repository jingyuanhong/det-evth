#!/usr/bin/env python3
"""
Color Analyzer - Find the actual color of ECG waveform

This tool samples colors from the ECG image to determine
the exact RGB/HSV values of the waveform.
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_ecg_colors(pdf_path):
    """Analyze actual colors in ECG PDF"""
    from pdf2image import convert_from_path

    print("Loading PDF...")
    images = convert_from_path(pdf_path, dpi=300)
    pil_image = images[0]
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]
    print(f"Image: {width}x{height} pixels\n")

    # Sample colors from middle region (where ECG waveform likely is)
    # Based on diagnostic: strips at rows ~300-2300
    sample_regions = [
        (300, 500, "First strip area"),
        (1000, 1200, "Middle strip area"),
        (2000, 2200, "Last strip area")
    ]

    all_sampled_colors_bgr = []
    all_sampled_colors_hsv = []

    for y_start, y_end, name in sample_regions:
        print(f"Sampling from {name} (rows {y_start}-{y_end})...")

        # Sample center region
        sample = image[y_start:y_end, width//4:3*width//4, :]

        # Find pixels that are likely waveform (not white background)
        # Background is very bright (R,G,B all > 200)
        # Waveform is darker/colored
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        not_background = gray < 200

        # Get those pixels
        waveform_pixels_bgr = sample[not_background]

        if len(waveform_pixels_bgr) > 0:
            # Convert to HSV for analysis
            sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
            waveform_pixels_hsv = sample_hsv[not_background]

            all_sampled_colors_bgr.append(waveform_pixels_bgr)
            all_sampled_colors_hsv.append(waveform_pixels_hsv)

            # Statistics
            mean_bgr = np.mean(waveform_pixels_bgr, axis=0)
            std_bgr = np.std(waveform_pixels_bgr, axis=0)

            mean_hsv = np.mean(waveform_pixels_hsv, axis=0)
            std_hsv = np.std(waveform_pixels_hsv, axis=0)

            print(f"  BGR: B={mean_bgr[0]:.1f}±{std_bgr[0]:.1f}, "
                  f"G={mean_bgr[1]:.1f}±{std_bgr[1]:.1f}, "
                  f"R={mean_bgr[2]:.1f}±{std_bgr[2]:.1f}")
            print(f"  HSV: H={mean_hsv[0]:.1f}±{std_hsv[0]:.1f}, "
                  f"S={mean_hsv[1]:.1f}±{std_hsv[1]:.1f}, "
                  f"V={mean_hsv[2]:.1f}±{std_hsv[2]:.1f}")
            print(f"  Sampled {len(waveform_pixels_bgr)} pixels\n")

    # Combine all samples
    if len(all_sampled_colors_bgr) > 0:
        all_bgr = np.vstack(all_sampled_colors_bgr)
        all_hsv = np.vstack(all_sampled_colors_hsv)

        print("=" * 70)
        print("OVERALL WAVEFORM COLOR ANALYSIS")
        print("=" * 70)

        mean_bgr = np.mean(all_bgr, axis=0)
        std_bgr = np.std(all_bgr, axis=0)
        median_bgr = np.median(all_bgr, axis=0)

        mean_hsv = np.mean(all_hsv, axis=0)
        std_hsv = np.std(all_hsv, axis=0)
        median_hsv = np.median(all_hsv, axis=0)

        print(f"\nBGR Statistics:")
        print(f"  Mean:   B={mean_bgr[0]:.1f}, G={mean_bgr[1]:.1f}, R={mean_bgr[2]:.1f}")
        print(f"  Median: B={median_bgr[0]:.1f}, G={median_bgr[1]:.1f}, R={median_bgr[2]:.1f}")
        print(f"  Std:    B={std_bgr[0]:.1f}, G={std_bgr[1]:.1f}, R={std_bgr[2]:.1f}")

        print(f"\nHSV Statistics:")
        print(f"  Mean:   H={mean_hsv[0]:.1f}°, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")
        print(f"  Median: H={median_hsv[0]:.1f}°, S={median_hsv[1]:.1f}, V={median_hsv[2]:.1f}")
        print(f"  Std:    H={std_hsv[0]:.1f}°, S={std_hsv[1]:.1f}, V={std_hsv[2]:.1f}")

        # Determine color description
        h = mean_hsv[0]
        s = mean_hsv[1]
        v = mean_hsv[2]

        if s < 50:
            color_name = "Gray/Low saturation"
        elif h < 10 or h > 170:
            if s < 100:
                color_name = "Pink/Light Red"
            else:
                color_name = "Red"
        elif 10 <= h < 30:
            color_name = "Orange/Coral"
        else:
            color_name = f"Other (H={h:.0f}°)"

        print(f"\nColor Description: {color_name}")

        print("\n" + "=" * 70)
        print("RECOMMENDED COLOR DETECTION THRESHOLDS")
        print("=" * 70)

        # Calculate thresholds with some margin
        b_low, b_high = max(0, mean_bgr[0] - 2*std_bgr[0]), min(255, mean_bgr[0] + 2*std_bgr[0])
        g_low, g_high = max(0, mean_bgr[1] - 2*std_bgr[1]), min(255, mean_bgr[1] + 2*std_bgr[1])
        r_low, r_high = max(0, mean_bgr[2] - 2*std_bgr[2]), min(255, mean_bgr[2] + 2*std_bgr[2])

        h_low, h_high = max(0, mean_hsv[0] - 2*std_hsv[0]), min(180, mean_hsv[0] + 2*std_hsv[0])
        s_low, s_high = max(0, mean_hsv[1] - 2*std_hsv[1]), min(255, mean_hsv[1] + 2*std_hsv[1])
        v_low, v_high = max(0, mean_hsv[2] - 2*std_hsv[2]), min(255, mean_hsv[2] + 2*std_hsv[2])

        print(f"\nBGR Range (mean ± 2*std):")
        print(f"  B: [{b_low:.0f}, {b_high:.0f}]")
        print(f"  G: [{g_low:.0f}, {g_high:.0f}]")
        print(f"  R: [{r_low:.0f}, {r_high:.0f}]")

        print(f"\nHSV Range (mean ± 2*std):")
        print(f"  H: [{h_low:.0f}, {h_high:.0f}]")
        print(f"  S: [{s_low:.0f}, {s_high:.0f}]")
        print(f"  V: [{v_low:.0f}, {v_high:.0f}]")

        print("\nRecommended Python code:")
        print("```python")
        print(f"# BGR method")
        print(f"b, g, r = cv2.split(image)")
        print(f"mask_bgr = ((b >= {b_low:.0f}) & (b <= {b_high:.0f}) &")
        print(f"            (g >= {g_low:.0f}) & (g <= {g_high:.0f}) &")
        print(f"            (r >= {r_low:.0f}) & (r <= {r_high:.0f})).astype(np.uint8) * 255")
        print()
        print(f"# HSV method")
        print(f"hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)")
        print(f"lower = np.array([{max(0, h_low):.0f}, {max(0, s_low):.0f}, {max(0, v_low):.0f}])")
        print(f"upper = np.array([{min(180, h_high):.0f}, {min(255, s_high):.0f}, {min(255, v_high):.0f}])")
        print(f"mask_hsv = cv2.inRange(hsv, lower, upper)")
        print("```")

        # Create visualization
        create_color_visualization(all_bgr, all_hsv, mean_bgr, mean_hsv)

    else:
        print("No waveform pixels found!")


def create_color_visualization(bgr_pixels, hsv_pixels, mean_bgr, mean_hsv):
    """Create color distribution visualization"""

    Path("output/debug").mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))

    # BGR histograms
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        ax = plt.subplot(2, 3, i+1)
        ax.hist(bgr_pixels[:, i], bins=50, color=col, alpha=0.7, edgecolor='black')
        ax.axvline(mean_bgr[i], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_bgr[i]:.1f}')
        ax.set_xlabel(f'{col.upper()} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col.upper()} Channel Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # HSV histograms
    hsv_names = ['Hue', 'Saturation', 'Value']
    for i, name in enumerate(hsv_names):
        ax = plt.subplot(2, 3, i+4)
        ax.hist(hsv_pixels[:, i], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(mean_hsv[i], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_hsv[i]:.1f}')
        ax.set_xlabel(f'{name} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/debug/color_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: output/debug/color_analysis.png")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_colors.py <ecg.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    analyze_ecg_colors(pdf_path)


if __name__ == "__main__":
    main()
