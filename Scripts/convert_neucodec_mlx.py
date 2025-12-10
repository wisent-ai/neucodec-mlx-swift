#!/usr/bin/env python3
"""
Convert NeuCodec PyTorch weights to MLX-compatible safetensors format.

Extracts decoder-only weights (generator.*) and remaps tensor names.
Handles weight normalization by computing the actual weights: weight = weight_g * weight_v / ||weight_v||
"""

import torch
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file
import json


def weight_norm_to_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Convert weight normalization parameters to actual weight matrix.

    PyTorch weight normalization: weight = g * v / ||v||
    where g is the magnitude and v is the direction.
    """
    norm_dims = tuple(range(1, weight_v.dim()))
    v_norm = torch.norm(weight_v, dim=norm_dims, keepdim=True)
    return weight_g * weight_v / v_norm


def convert_neucodec_to_mlx(
    input_path: Path,
    output_path: Path,
    config_path: Path
):
    """Convert NeuCodec PyTorch checkpoint to MLX safetensors."""

    print(f"Loading PyTorch checkpoint from {input_path}")
    state_dict = torch.load(input_path, map_location="cpu")

    print(f"Total tensors: {len(state_dict)}")

    # Collect all decoder weights
    decoder_weights = {}

    for k, v in state_dict.items():
        if k.startswith("generator."):
            decoder_weights[k] = v

    # Also include fc_post_a
    if "fc_post_a.weight" in state_dict:
        decoder_weights["fc_post_a.weight"] = state_dict["fc_post_a.weight"]
        decoder_weights["fc_post_a.bias"] = state_dict["fc_post_a.bias"]

    print(f"Decoder tensors: {len(decoder_weights)}")

    mlx_weights = {}
    processed_keys = set()

    # First pass: handle weight normalization (weight_g + weight_v -> weight)
    for key in list(decoder_weights.keys()):
        if key in processed_keys:
            continue

        if key.endswith(".weight_g"):
            base_key = key[:-2]  # Remove "_g" to get base.weight
            weight_v_key = base_key + "_v"

            if weight_v_key in decoder_weights:
                weight_g = decoder_weights[key]
                weight_v = decoder_weights[weight_v_key]
                actual_weight = weight_norm_to_weight(weight_g, weight_v)

                output_key = base_key.replace("generator.", "")
                mlx_weights[output_key] = actual_weight.numpy()

                processed_keys.add(key)
                processed_keys.add(weight_v_key)

                # Also mark plain weight as processed if it exists (some models have both)
                plain_weight_key = key.replace(".weight_g", ".weight")
                if plain_weight_key in decoder_weights:
                    processed_keys.add(plain_weight_key)

    # Second pass: handle remaining weights (biases, non-normalized weights, etc.)
    for key, tensor in decoder_weights.items():
        if key in processed_keys:
            continue

        # Skip weight_v keys (already processed with weight_g)
        if key.endswith(".weight_v"):
            continue

        # Skip plain .weight if there's a corresponding weight_g (we use normalized version)
        if key.endswith(".weight"):
            weight_g_key = key.replace(".weight", ".weight_g")
            if weight_g_key in decoder_weights:
                continue

        output_key = key.replace("generator.", "")
        mlx_weights[output_key] = tensor.numpy()

    # Rename keys to MLX-friendly format
    renamed_weights = {}
    for key, value in mlx_weights.items():
        new_key = key

        new_key = new_key.replace("backbone.transformers.", "transformers.")
        new_key = new_key.replace("backbone.prior_net.", "priorNet.")
        new_key = new_key.replace("backbone.post_net.", "postNet.")
        new_key = new_key.replace("backbone.embed.", "embed.")
        new_key = new_key.replace("backbone.final_layer_norm.", "finalNorm.")
        new_key = new_key.replace(".att.", ".attention.")
        new_key = new_key.replace(".att_norm.", ".attNorm.")
        new_key = new_key.replace(".ffn_norm.", ".ffnNorm.")
        new_key = new_key.replace("fc_post_a.", "fcPostA.")
        new_key = new_key.replace("quantizer.project_in.", "quantizer.projectIn.")
        new_key = new_key.replace("quantizer.project_out.", "quantizer.projectOut.")
        new_key = new_key.replace(".c_attn.", ".cAttn.")
        new_key = new_key.replace(".c_proj.", ".cProj.")

        renamed_weights[new_key] = value

    # Print summary
    print(f"\nConverted {len(renamed_weights)} weights:")
    for key in sorted(renamed_weights.keys()):
        print(f"  {key}: {renamed_weights[key].shape}")

    # Calculate total parameters
    total_params = sum(v.size for v in renamed_weights.values())
    print(f"\nTotal parameters: {total_params:,}")

    # Save as safetensors
    print(f"\nSaving to {output_path}")
    save_file(renamed_weights, str(output_path))

    # Determine config from weights
    head_out_shape = renamed_weights.get("head.out.weight", np.zeros((0,))).shape
    n_fft = (head_out_shape[0] - 2) if len(head_out_shape) > 0 else 1920

    config = {
        "hidden_dim": 1024,
        "transformer_depth": 12,
        "attention_heads": 16,
        "head_dim": 64,
        "mlp_expansion": 4,
        "n_fft": n_fft,
        "hop_length": n_fft // 4,
        "sample_rate": 24000,
        "fsq_levels": [4, 4, 4, 4, 4, 4, 4, 4],
    }

    print(f"\nConfig: {config}")
    print(f"Saving config to {config_path}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone!")
    return renamed_weights


if __name__ == "__main__":
    model_path = Path.home() / ".cache/huggingface/hub/models--neuphonic--neucodec/snapshots/c92ba97d538f2a0baa9118c21ea5de4cdad4e02a/pytorch_model.bin"

    output_dir = Path(__file__).parent.parent / "Wisent" / "Resources" / "neucodec"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "neucodec_decoder.safetensors"
    config_path = output_dir / "neucodec_config.json"

    convert_neucodec_to_mlx(model_path, output_path, config_path)
