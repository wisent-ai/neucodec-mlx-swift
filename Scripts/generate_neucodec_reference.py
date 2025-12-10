#!/usr/bin/env python3
"""
Generate reference audio using the actual NeuCodec library.
This uses the exact same decoder that NeuTTS uses.
"""

import torch
import numpy as np
from pathlib import Path
import json

# Import the actual NeuCodec decoder
from neucodec import NeuCodec


def main():
    print("=== Generate Reference Audio using actual NeuCodec ===\n")

    output_dir = Path(__file__).parent / "debug_outputs"
    output_dir.mkdir(exist_ok=True)

    # Load the actual NeuCodec decoder (same as NeuTTS uses)
    print("Loading NeuCodec from 'neuphonic/neucodec'...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval()
    print("NeuCodec loaded successfully!")

    # Test codes - same as in Swift tests
    test_cases = {
        "test_0": [56422, 15795, 860, 38158, 62570, 54343, 44732, 11284, 54886, 6265],
        "test_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "test_2": [65535, 65534, 65533, 65532, 65531],
        "test_3": [32768, 32768, 32768, 32768, 32768],
    }

    results = {}

    for name, codes_list in test_cases.items():
        print(f"\n--- {name}: {codes_list[:5]}... ---")

        # Create tensor in the format expected by NeuCodec: (batch, 1, num_codes)
        codes = torch.tensor(codes_list, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        print(f"  Input shape: {codes.shape}")

        with torch.no_grad():
            # Use the actual decode_code method (same as NeuTTS _decode)
            audio = codec.decode_code(codes)
            audio = audio.squeeze().cpu().numpy()

        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio length: {len(audio)}")
        print(f"  Mean: {audio.mean():.6f}")
        print(f"  Std: {audio.std():.6f}")
        print(f"  Min: {audio.min():.6f}")
        print(f"  Max: {audio.max():.6f}")
        print(f"  First 20: {audio[:20].tolist()}")

        # Save audio
        np.save(output_dir / f"neucodec_reference_{name}.npy", audio)

        results[name] = {
            "codes": codes_list,
            "audio_length": len(audio),
            "mean": float(audio.mean()),
            "std": float(audio.std()),
            "min": float(audio.min()),
            "max": float(audio.max()),
            "first_50": audio[:50].tolist(),
            "last_20": audio[-20:].tolist(),
        }

    # Save results
    with open(output_dir / "neucodec_reference_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nReference audio saved to {output_dir}")
    print("Files generated:")
    for name in test_cases:
        print(f"  - neucodec_reference_{name}.npy")
    print("  - neucodec_reference_stats.json")


if __name__ == "__main__":
    main()
