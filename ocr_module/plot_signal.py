#!/usr/bin/env python3
"""
Simple ECG signal plotter

Usage:
    python plot_signal.py
    python plot_signal.py output/extracted_ecg_signal.npy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_ecg_signal(signal_path='output/extracted_ecg_signal.npy',
                    time_path='output/extracted_ecg_time.npy'):
    """Plot extracted ECG signal"""

    # Check if files exist
    if not Path(signal_path).exists():
        print(f"Error: Signal file not found: {signal_path}")
        print("\nMake sure you've run the OCR extraction first:")
        print("  python demo_test.py data/sample_ecg.pdf")
        return

    if not Path(time_path).exists():
        print(f"Error: Time file not found: {time_path}")
        return

    # Load data
    print(f"Loading signal from: {signal_path}")
    signal = np.load(signal_path)
    time = np.load(time_path)

    # Print info
    print(f"\nSignal Information:")
    print(f"  Total samples: {len(signal):,}")
    print(f"  Duration: {time[-1]:.2f} seconds")
    print(f"  Sampling rate: {len(signal)/time[-1]:.1f} Hz")
    print(f"  Voltage range: [{signal.min():.3f}, {signal.max():.3f}] mV")
    print(f"  Mean: {signal.mean():.3f} mV")
    print(f"  Std: {signal.std():.3f} mV")
    print()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # Plot 1: Full signal (30 seconds)
    ax1.plot(time, signal, 'r-', linewidth=0.8)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Voltage (mV)', fontsize=12)
    ax1.set_title('Complete ECG Signal (30 seconds)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed view (first 5 seconds)
    mask = time <= 5.0
    ax2.plot(time[mask], signal[mask], 'r-', linewidth=1.2)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Voltage (mV)', fontsize=12)
    ax2.set_title('Zoomed View - First 5 Seconds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(signal_path).parent / 'ecg_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Show plot
    print("\nDisplaying plot... (close window to exit)")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        signal_path = sys.argv[1]
        # Try to find corresponding time file
        time_path = signal_path.replace('signal', 'time')
        plot_ecg_signal(signal_path, time_path)
    else:
        plot_ecg_signal()
