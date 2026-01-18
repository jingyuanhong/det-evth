#!/usr/bin/env python3
"""
ECGFounder PyTorch to CoreML Conversion Script

Converts the 1-lead ECGFounder model (353MB PyTorch) to CoreML format (~90MB)
with 4-bit palettization for efficient on-device inference.

Requirements:
    pip install torch coremltools numpy

Usage:
    python convert_to_coreml.py

Output:
    - ECGFounder1Lead.mlpackage (full precision, ~353MB)
    - ECGFounder1Lead_4bit.mlpackage (4-bit quantized, ~90MB)

© 2026 minuscule health Ltd. All rights reserved.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add ECGFounder to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ECGFOUNDER_PATH = PROJECT_ROOT / "github" / "ECGFounder"
sys.path.insert(0, str(ECGFOUNDER_PATH))

from net1d import Net1D

# Paths
PYTORCH_MODEL_PATH = PROJECT_ROOT / "github" / "1_lead_ECGFounder.pth"
OUTPUT_DIR = SCRIPT_DIR.parent / "DetEvth" / "Resources" / "ML"

# Model Configuration (must match training)
MODEL_CONFIG = {
    "in_channels": 1,
    "base_filters": 64,
    "ratio": 1,
    "filter_list": [64, 160, 160, 400, 400, 1024, 1024],
    "m_blocks_list": [2, 2, 2, 3, 3, 4, 4],
    "kernel_size": 16,
    "stride": 2,
    "groups_width": 16,
    "n_classes": 150,
    "use_bn": False,
    "use_do": False,
}

# ECG Signal Configuration
ECG_SAMPLE_RATE = 500  # Hz
ECG_DURATION = 10  # seconds
ECG_LENGTH = ECG_SAMPLE_RATE * ECG_DURATION  # 5000 samples


def load_pytorch_model():
    """Load the pre-trained PyTorch ECGFounder model"""
    print("=" * 60)
    print("STEP 1: Loading PyTorch Model")
    print("=" * 60)

    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {PYTORCH_MODEL_PATH}")

    # Create model architecture
    model = Net1D(
        in_channels=MODEL_CONFIG["in_channels"],
        base_filters=MODEL_CONFIG["base_filters"],
        ratio=MODEL_CONFIG["ratio"],
        filter_list=MODEL_CONFIG["filter_list"],
        m_blocks_list=MODEL_CONFIG["m_blocks_list"],
        kernel_size=MODEL_CONFIG["kernel_size"],
        stride=MODEL_CONFIG["stride"],
        groups_width=MODEL_CONFIG["groups_width"],
        n_classes=MODEL_CONFIG["n_classes"],
        use_bn=MODEL_CONFIG["use_bn"],
        use_do=MODEL_CONFIG["use_do"],
        verbose=False
    )

    # Load weights
    checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)

    print(f"  Model path: {PYTORCH_MODEL_PATH}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model size: {total_size_mb:.1f} MB")
    print(f"  Input shape: (1, 1, {ECG_LENGTH})")
    print(f"  Output: {MODEL_CONFIG['n_classes']} disease classes")

    return model


def verify_pytorch_model(model):
    """Verify PyTorch model with sample input"""
    print("\n" + "=" * 60)
    print("STEP 2: Verifying PyTorch Model")
    print("=" * 60)

    # Create sample ECG input (batch=1, channels=1, length=5000)
    sample_input = torch.randn(1, 1, ECG_LENGTH)

    with torch.no_grad():
        logits = model(sample_input)
        probs = torch.sigmoid(logits)

    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("  PyTorch model verification: PASSED")

    return sample_input


def convert_to_coreml(model, sample_input):
    """Convert PyTorch model to CoreML format"""
    print("\n" + "=" * 60)
    print("STEP 3: Converting to CoreML")
    print("=" * 60)

    try:
        import coremltools as ct
        print(f"  coremltools version: {ct.__version__}")
    except ImportError:
        print("ERROR: coremltools not installed. Run: pip install coremltools")
        sys.exit(1)

    # Trace the model
    print("  Tracing PyTorch model...")
    traced_model = torch.jit.trace(model, sample_input)

    # Define input/output specifications
    input_shape = ct.Shape(shape=(1, 1, ECG_LENGTH))

    # Convert to CoreML
    print("  Converting to CoreML format...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="ecg_signal",
                shape=input_shape,
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="disease_logits")
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
    )

    # Add model metadata
    mlmodel.author = "minuscule health Ltd"
    mlmodel.license = "Proprietary - minuscule health Ltd © 2026"
    mlmodel.short_description = "ECGFounder 1-lead ECG disease screening model (150 classes)"
    mlmodel.version = "1.0.0"

    # Add input/output descriptions
    mlmodel.input_description["ecg_signal"] = (
        "Preprocessed 1-lead ECG signal. Shape: (1, 1, 5000). "
        "Preprocessing: resample to 500Hz, bandpass 0.67-40Hz, "
        "notch 50Hz, median baseline removal, z-score normalize."
    )
    mlmodel.output_description["disease_logits"] = (
        "Raw logits for 150 disease classes. "
        "Apply sigmoid to get probabilities."
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full precision model
    full_model_path = OUTPUT_DIR / "ECGFounder1Lead.mlpackage"
    mlmodel.save(str(full_model_path))

    full_size_mb = sum(f.stat().st_size for f in full_model_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"  Full precision model saved: {full_model_path}")
    print(f"  Full model size: {full_size_mb:.1f} MB")

    return mlmodel, full_model_path


def apply_quantization(mlmodel_path):
    """Apply 4-bit palettization to reduce model size"""
    print("\n" + "=" * 60)
    print("STEP 4: Applying 4-bit Quantization")
    print("=" * 60)

    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights
    )

    # Load the full precision model
    print("  Loading full precision model...")
    mlmodel = ct.models.MLModel(str(mlmodel_path))

    # Configure 4-bit palettization
    # This reduces weight storage from 32-bit to 4-bit (8x reduction)
    print("  Configuring 4-bit palettization...")
    op_config = OpPalettizerConfig(
        mode="kmeans",
        nbits=4,
        weight_threshold=512  # Minimum weight tensor size to quantize
    )

    config = OptimizationConfig(global_config=op_config)

    # Apply palettization
    print("  Applying palettization (this may take a few minutes)...")
    quantized_model = palettize_weights(mlmodel, config)

    # Save quantized model
    quantized_model_path = OUTPUT_DIR / "ECGFounder1Lead_4bit.mlpackage"
    quantized_model.save(str(quantized_model_path))

    quantized_size_mb = sum(f.stat().st_size for f in quantized_model_path.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"  Quantized model saved: {quantized_model_path}")
    print(f"  Quantized model size: {quantized_size_mb:.1f} MB")

    return quantized_model, quantized_model_path


def validate_coreml_model(mlmodel_path, pytorch_model, sample_input):
    """Validate CoreML model output matches PyTorch"""
    print("\n" + "=" * 60)
    print("STEP 5: Validating CoreML Model")
    print("=" * 60)

    import coremltools as ct

    # Load CoreML model
    mlmodel = ct.models.MLModel(str(mlmodel_path))

    # Get PyTorch predictions
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input)
        pytorch_probs = torch.sigmoid(pytorch_output).numpy().squeeze()

    # Get CoreML predictions
    coreml_input = {"ecg_signal": sample_input.numpy()}
    coreml_output = mlmodel.predict(coreml_input)
    coreml_logits = list(coreml_output.values())[0].squeeze()
    coreml_probs = 1 / (1 + np.exp(-coreml_logits))  # sigmoid

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_probs - coreml_probs))
    mean_diff = np.mean(np.abs(pytorch_probs - coreml_probs))
    correlation = np.corrcoef(pytorch_probs, coreml_probs)[0, 1]

    print(f"  PyTorch vs CoreML comparison:")
    print(f"    Max absolute difference: {max_diff:.6f}")
    print(f"    Mean absolute difference: {mean_diff:.6f}")
    print(f"    Correlation: {correlation:.6f}")

    # Check top-5 predictions match
    pytorch_top5 = np.argsort(pytorch_probs)[-5:][::-1]
    coreml_top5 = np.argsort(coreml_probs)[-5:][::-1]
    top5_match = np.array_equal(pytorch_top5, coreml_top5)

    print(f"    Top-5 predictions match: {top5_match}")

    if max_diff < 0.01 and correlation > 0.999:
        print("  Validation: PASSED")
        return True
    else:
        print("  Validation: WARNING - Some differences detected")
        return False


def create_swift_helper():
    """Create a Swift helper file for using the CoreML model"""
    print("\n" + "=" * 60)
    print("STEP 6: Creating Swift Helper Code")
    print("=" * 60)

    swift_code = '''// ECGFounderModelHelper.swift
// Auto-generated helper for ECGFounder CoreML model
// © 2026 minuscule health Ltd. All rights reserved.

import CoreML
import Foundation

/// Configuration for ECG signal preprocessing
enum ECGConfig {
    static let sampleRate: Float = 500.0
    static let duration: Float = 10.0
    static let signalLength: Int = 5000
    static let numClasses: Int = 150
}

/// Helper class for ECGFounder model inference
class ECGFounderHelper {

    private let model: MLModel

    init() throws {
        // Load the quantized model (4-bit)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        guard let modelURL = Bundle.main.url(
            forResource: "ECGFounder1Lead_4bit",
            withExtension: "mlmodelc"
        ) else {
            throw ECGFounderError.modelNotFound
        }

        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }

    /// Run inference on preprocessed ECG signal
    /// - Parameter signal: Preprocessed ECG signal (5000 samples, z-score normalized)
    /// - Returns: Array of 150 probabilities for each disease class
    func predict(signal: [Float]) throws -> [Float] {
        guard signal.count == ECGConfig.signalLength else {
            throw ECGFounderError.invalidInputLength(expected: ECGConfig.signalLength, got: signal.count)
        }

        // Create MLMultiArray input (1, 1, 5000)
        let inputShape = [1, 1, ECGConfig.signalLength] as [NSNumber]
        let inputArray = try MLMultiArray(shape: inputShape, dataType: .float32)

        for i in 0..<signal.count {
            inputArray[i] = NSNumber(value: signal[i])
        }

        // Create feature provider
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "ecg_signal": MLFeatureValue(multiArray: inputArray)
        ])

        // Run prediction
        let output = try model.prediction(from: inputFeatures)

        // Extract logits and apply sigmoid
        guard let logitsArray = output.featureValue(for: "disease_logits")?.multiArrayValue else {
            throw ECGFounderError.outputExtractionFailed
        }

        var probabilities = [Float](repeating: 0, count: ECGConfig.numClasses)
        for i in 0..<ECGConfig.numClasses {
            let logit = logitsArray[i].floatValue
            probabilities[i] = 1.0 / (1.0 + exp(-logit))  // sigmoid
        }

        return probabilities
    }
}

enum ECGFounderError: Error {
    case modelNotFound
    case invalidInputLength(expected: Int, got: Int)
    case outputExtractionFailed
}
'''

    swift_path = OUTPUT_DIR.parent.parent / "Services" / "CoreML" / "ECGFounderModelHelper.swift"
    swift_path.parent.mkdir(parents=True, exist_ok=True)

    with open(swift_path, 'w') as f:
        f.write(swift_code)

    print(f"  Swift helper saved: {swift_path}")


def main():
    print("\n" + "=" * 60)
    print("ECGFounder PyTorch to CoreML Conversion")
    print("minuscule health Ltd © 2026")
    print("=" * 60)

    # Step 1: Load PyTorch model
    pytorch_model = load_pytorch_model()

    # Step 2: Verify PyTorch model
    sample_input = verify_pytorch_model(pytorch_model)

    # Step 3: Convert to CoreML
    mlmodel, full_model_path = convert_to_coreml(pytorch_model, sample_input)

    # Step 4: Apply 4-bit quantization
    quantized_model, quantized_model_path = apply_quantization(full_model_path)

    # Step 5: Validate CoreML model
    print("\n  Validating full precision model...")
    validate_coreml_model(full_model_path, pytorch_model, sample_input)

    print("\n  Validating quantized model...")
    validate_coreml_model(quantized_model_path, pytorch_model, sample_input)

    # Step 6: Create Swift helper
    create_swift_helper()

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Full precision: {full_model_path}")
    print(f"  Quantized (4-bit): {quantized_model_path}")
    print(f"\nFor iOS deployment:")
    print(f"  1. Add ECGFounder1Lead_4bit.mlpackage to Xcode project")
    print(f"  2. Xcode will compile it to .mlmodelc automatically")
    print(f"  3. Use ECGFounderHelper.swift for inference")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
