#!/usr/bin/env python3
"""
Generate reference audio from Python NeuCodec decoder for comparison with Swift.
Uses the actual NeuCodec model to generate audio that can be compared sample-by-sample.
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys

# Try to import neucodec directly from its location
NEUCODEC_PATH = Path.home() / ".pyenv/versions/3.11.11/lib/python3.11/site-packages"
sys.path.insert(0, str(NEUCODEC_PATH))

# Import only what we need from neucodec without triggering full transformers import
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--neuphonic--neucodec/snapshots/c92ba97d538f2a0baa9118c21ea5de4cdad4e02a/pytorch_model.bin"


def load_generator_directly():
    """Load just the generator/decoder part of NeuCodec"""
    import torch.nn as nn
    from torch.nn import functional as F

    # Load state dict
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # Build the model architecture matching the weights
    # We need: quantizer, fc_post_a, backbone, head

    class FSQDecoder(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.levels = [4, 4, 4, 4, 4, 4, 4, 4]
            self.basis = torch.tensor([1, 4, 16, 64, 256, 1024, 4096, 16384])
            self.project_out = nn.Linear(8, 2048)
            self.project_out.weight.data = state_dict["generator.quantizer.project_out.weight"]
            self.project_out.bias.data = state_dict["generator.quantizer.project_out.bias"]

        def forward(self, indices):
            # indices: (B, 1, T) -> level_indices: (B, T, 8)
            indices = indices.squeeze(1)  # (B, T)
            indices_expanded = indices.unsqueeze(-1)  # (B, T, 1)

            levels = torch.tensor(self.levels)
            level_indices = (indices_expanded // self.basis) % levels  # (B, T, 8)

            half_levels = (levels.float() - 1) / 2
            scaled = (level_indices.float() - half_levels) / half_levels  # (B, T, 8)

            return self.project_out(scaled)  # (B, T, 2048)

    class ISTFT(nn.Module):
        def __init__(self, n_fft=1920, hop_length=480):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = n_fft
            window = torch.hann_window(n_fft, periodic=True)
            self.register_buffer("window", window)

        def forward(self, spec):
            # spec: (B, N, T) complex
            pad = (self.win_length - self.hop_length) // 2

            B, N, T = spec.shape
            ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
            ifft = ifft * self.window[None, :, None]

            output_size = (T - 1) * self.hop_length + self.win_length
            y = F.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )[:, 0, 0, pad:-pad]

            window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
            window_envelope = F.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            ).squeeze()[pad:-pad]

            y = y / window_envelope
            return y

    class ResnetBlock(nn.Module):
        def __init__(self, channels, prefix, state_dict):
            super().__init__()
            self.norm1 = nn.GroupNorm(32, channels, eps=1e-6)
            self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
            self.norm2 = nn.GroupNorm(32, channels, eps=1e-6)
            self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

            self.norm1.weight.data = state_dict[f"{prefix}.norm1.weight"]
            self.norm1.bias.data = state_dict[f"{prefix}.norm1.bias"]
            self.conv1.weight.data = state_dict[f"{prefix}.conv1.weight"]
            self.conv1.bias.data = state_dict[f"{prefix}.conv1.bias"]
            self.norm2.weight.data = state_dict[f"{prefix}.norm2.weight"]
            self.norm2.bias.data = state_dict[f"{prefix}.norm2.bias"]
            self.conv2.weight.data = state_dict[f"{prefix}.conv2.weight"]
            self.conv2.bias.data = state_dict[f"{prefix}.conv2.bias"]

        def forward(self, x):
            h = self.norm1(x)
            h = F.silu(h)
            h = self.conv1(h)
            h = self.norm2(h)
            h = F.silu(h)
            h = self.conv2(h)
            return x + h

    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, prefix, state_dict):
            super().__init__()
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads

            self.att_norm = nn.RMSNorm(dim, eps=1e-6)
            self.ffn_norm = nn.RMSNorm(dim, eps=1e-6)

            # GPT-2 style naming: c_attn and c_proj
            self.c_attn = nn.Linear(dim, dim * 3, bias=False)
            self.c_proj = nn.Linear(dim, dim, bias=False)

            # Simple MLP: fc1 -> SiLU -> fc2 (matches actual NeuCodec bs_roformer5.py)
            self.mlp_fc1 = nn.Linear(dim, dim * 4, bias=False)
            self.mlp_fc2 = nn.Linear(dim * 4, dim, bias=False)

            # Load weights
            self.att_norm.weight.data = state_dict[f"{prefix}.att_norm.weight"]
            self.ffn_norm.weight.data = state_dict[f"{prefix}.ffn_norm.weight"]
            self.c_attn.weight.data = state_dict[f"{prefix}.att.c_attn.weight"]
            self.c_proj.weight.data = state_dict[f"{prefix}.att.c_proj.weight"]
            self.mlp_fc1.weight.data = state_dict[f"{prefix}.mlp.fc1.weight"]
            self.mlp_fc2.weight.data = state_dict[f"{prefix}.mlp.fc2.weight"]

            # RoPE cache
            self._rope_cache = {}

        def _get_rope(self, seq_len, device):
            if seq_len not in self._rope_cache:
                positions = torch.arange(seq_len, device=device)
                dim_half = self.head_dim // 2
                freqs = 10000.0 ** (-torch.arange(0, dim_half, device=device).float() / dim_half)
                angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
                cos = torch.cos(angles)
                sin = torch.sin(angles)
                self._rope_cache[seq_len] = (cos, sin)
            return self._rope_cache[seq_len]

        def _apply_rope(self, x, cos, sin):
            # x: (B, H, T, D)
            # Interleaved pairs approach (matching torchtune RotaryPositionalEmbeddings)
            # Reshape to extract pairs: [..., D] -> [..., D/2, 2]
            B, H, T, D = x.shape
            x_reshaped = x.reshape(B, H, T, D // 2, 2)

            # cos/sin: (T, D/2) -> (1, 1, T, D/2, 1)
            cos = cos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            sin = sin.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

            # Apply rotation: out[0] = x[0]*cos - x[1]*sin, out[1] = x[1]*cos + x[0]*sin
            x0 = x_reshaped[..., 0:1]
            x1 = x_reshaped[..., 1:2]
            rotated = torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)

            return rotated.reshape(B, H, T, D)

        def forward(self, x):
            B, T, _ = x.shape

            # Attention
            h = self.att_norm(x)
            qkv = self.c_attn(h)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            cos, sin = self._get_rope(T, x.device)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

            out = out.transpose(1, 2).reshape(B, T, self.dim)
            out = self.c_proj(out)
            x = x + out

            # FFN: fc1 -> SiLU -> fc2 (matches actual NeuCodec bs_roformer5.py)
            h = self.ffn_norm(x)
            h = self.mlp_fc1(h)
            h = F.silu(h)
            h = self.mlp_fc2(h)
            x = x + h

            return x

    class NeuCodecGenerator(nn.Module):
        def __init__(self, state_dict):
            super().__init__()

            self.quantizer = FSQDecoder(state_dict)
            self.fc_post_a = nn.Linear(2048, 1024)
            self.fc_post_a.weight.data = state_dict["fc_post_a.weight"]
            self.fc_post_a.bias.data = state_dict["fc_post_a.bias"]

            # Backbone
            self.embed = nn.Conv1d(1024, 1024, 7, padding=3)
            self.embed.weight.data = state_dict["generator.backbone.embed.weight"]
            self.embed.bias.data = state_dict["generator.backbone.embed.bias"]

            self.prior_blocks = nn.ModuleList([
                ResnetBlock(1024, f"generator.backbone.prior_net.{i}", state_dict)
                for i in range(2)
            ])

            self.transformers = nn.ModuleList([
                TransformerBlock(1024, 16, f"generator.backbone.transformers.{i}", state_dict)
                for i in range(12)
            ])

            self.final_norm = nn.LayerNorm(1024, eps=1e-6)
            self.final_norm.weight.data = state_dict["generator.backbone.final_layer_norm.weight"]
            self.final_norm.bias.data = state_dict["generator.backbone.final_layer_norm.bias"]

            self.post_blocks = nn.ModuleList([
                ResnetBlock(1024, f"generator.backbone.post_net.{i}", state_dict)
                for i in range(2)
            ])

            # Head
            self.head_out = nn.Linear(1024, 1922)
            self.head_out.weight.data = state_dict["generator.head.out.weight"]
            self.head_out.bias.data = state_dict["generator.head.out.bias"]

            self.istft = ISTFT()

        def decode(self, codes):
            # codes: (B, 1, T)
            h = self.quantizer(codes)  # (B, T, 2048)
            h = self.fc_post_a(h)  # (B, T, 1024)

            # Backbone
            h = h.transpose(1, 2)  # (B, 1024, T)
            h = self.embed(h)

            for block in self.prior_blocks:
                h = block(h)

            h = h.transpose(1, 2)  # (B, T, 1024)
            for transformer in self.transformers:
                h = transformer(h)
            h = h.transpose(1, 2)  # (B, 1024, T)

            for block in self.post_blocks:
                h = block(h)

            h = h.transpose(1, 2)  # (B, T, 1024)
            h = self.final_norm(h)

            # Head
            x_pred = self.head_out(h)  # (B, T, 1922)
            x_pred = x_pred.transpose(1, 2)  # (B, 1922, T)

            mag = x_pred[:, :961, :]
            phase = x_pred[:, 961:, :]

            mag = torch.exp(mag).clip(max=100.0)
            S = mag * (torch.cos(phase) + 1j * torch.sin(phase))

            audio = self.istft(S)
            return audio

    return NeuCodecGenerator(state_dict)


def main():
    print("=== Generate Reference Audio ===\n")

    output_dir = Path(__file__).parent / "debug_outputs"
    output_dir.mkdir(exist_ok=True)

    # Test codes - same as in Swift tests
    test_cases = {
        "test_0": [56422, 15795, 860, 38158, 62570, 54343, 44732, 11284, 54886, 6265],
        "test_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "test_2": [65535, 65534, 65533, 65532, 65531],
        "test_3": [32768, 32768, 32768, 32768, 32768],
    }

    print(f"Loading generator from {MODEL_PATH}")
    generator = load_generator_directly()
    generator.eval()
    print("Generator loaded successfully!")

    results = {}

    for name, codes_list in test_cases.items():
        print(f"\n--- {name}: {codes_list[:5]}... ---")

        codes = torch.tensor(codes_list, dtype=torch.long).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        with torch.no_grad():
            audio = generator.decode(codes)
            audio = audio.squeeze()

        audio_np = audio.numpy()

        print(f"  Audio length: {len(audio_np)}")
        print(f"  Mean: {audio_np.mean():.6f}")
        print(f"  Std: {audio_np.std():.6f}")
        print(f"  Min: {audio_np.min():.6f}")
        print(f"  Max: {audio_np.max():.6f}")
        print(f"  First 20: {audio_np[:20].tolist()}")

        # Save audio
        np.save(output_dir / f"reference_audio_{name}.npy", audio_np)

        results[name] = {
            "codes": codes_list,
            "audio_length": len(audio_np),
            "mean": float(audio_np.mean()),
            "std": float(audio_np.std()),
            "min": float(audio_np.min()),
            "max": float(audio_np.max()),
            "first_50": audio_np[:50].tolist(),
            "last_20": audio_np[-20:].tolist(),
        }

    # Save results
    with open(output_dir / "reference_audio_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nReference audio saved to {output_dir}")
    print("Files generated:")
    for name in test_cases:
        print(f"  - reference_audio_{name}.npy")
    print("  - reference_audio_stats.json")


if __name__ == "__main__":
    main()
