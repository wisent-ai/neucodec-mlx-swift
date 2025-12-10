# NeuCodec MLX Swift

A Swift implementation of the [NeuCodec](https://github.com/neuphonic/neucodec) neural audio codec decoder using Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework for GPU-accelerated inference on Apple Silicon.

## Features

- Decodes NeuCodec speech codes to 24kHz audio
- GPU-accelerated inference using MLX
- Sample-accurate output matching Python reference implementation
- iOS 16+ and macOS 14+ support

## Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/wisent-ai/neucodec-mlx-swift", from: "0.1.0")
]
```

Then add `NeuCodec` to your target dependencies:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "NeuCodec", package: "neucodec-mlx-swift")
    ]
)
```

## Usage

```swift
import NeuCodec

// Initialize decoder with weights file
let decoder = try NeuCodecMLXDecoder(modelPath: weightsURL)

// Decode speech codes to audio
let audioSamples = try decoder.decode(codes: speechCodes)

// audioSamples is [Float] at 24kHz sample rate
```

## Model Weights

The decoder requires pre-converted weights in safetensors format. The weights are **not included** in this package due to size.

### Converting Weights

Use the provided Python script to convert weights from the original NeuCodec model:

```bash
cd Scripts
python convert_neucodec_mlx.py --output neucodec_decoder.safetensors
```

This will download the original model from HuggingFace and convert it to MLX format.

### Requirements for conversion

```bash
pip install torch safetensors huggingface_hub
```

## Architecture

The decoder implements the following pipeline:

1. **FSQ Decode**: Integer codes → 2048-dim embeddings via Finite Scalar Quantization
2. **Linear Projection**: 2048 → 1024 dimensions
3. **VocosBackbone**: Conv1d + 2x ResNet + 12x Transformer + 2x ResNet + LayerNorm
4. **ISTFTHead**: Linear projection → magnitude/phase → Inverse STFT → audio

Key specifications:
- Sample rate: 24,000 Hz
- Hop length: 480 samples (20ms)
- STFT size: 1920
- Hidden dimension: 1024
- Transformer depth: 12 layers
- Attention heads: 16

## Testing

The package includes comparison tests that verify output matches the Python reference:

```bash
# Set model path
export NEUCODEC_MODEL_PATH=/path/to/neucodec_decoder.safetensors

# Run tests
swift test
```

Test data in `Tests/NeuCodecTests/Resources/` includes:
- `neucodec_debug.json`: Input speech codes for testing
- `python_decoder_output.npy`: Reference audio from Python implementation

## Scripts

- `convert_neucodec_mlx.py`: Converts PyTorch weights to MLX safetensors format
- `generate_reference_audio.py`: Standalone Python decoder for generating reference outputs
- `generate_neucodec_reference.py`: Uses official NeuCodec library to generate references

## License

Apache License 2.0 (same as NeuCodec)

## Acknowledgments

- [NeuCodec](https://github.com/neuphonic/neucodec) by Neuphonic
- [MLX Swift](https://github.com/ml-explore/mlx-swift) by Apple
