#!/usr/bin/env python3
"""Pure-NumPy reference for depthwise causal 1D convolution.

Math:
    out[b, d, t] = bias[d] + sum_{k=0..W-1} weight[d, k] * x[b, d, t - W + 1 + k]

with out-of-bounds inputs treated as 0 (causal left zero-pad of width W-1).
Vectorized via `np.pad` + a single per-tap loop (W iterations).

Usage:
    python scripts/cpu_reference.py --shape 0
    python scripts/cpu_reference.py --shape 2 --validate-against-pytorch
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# Test shapes copied inline from kernels/causal_conv1d/task.yml (5 entries).
TEST_SHAPES = [
    {"B": 1, "D": 64,  "S": 64,  "W": 4, "seed": 4242},
    {"B": 2, "D": 128, "S": 128, "W": 4, "seed": 5236},
    {"B": 1, "D": 256, "S": 256, "W": 3, "seed": 1001},
    {"B": 1, "D": 128, "S": 64,  "W": 8, "seed": 5531},
    {"B": 4, "D": 64,  "S": 128, "W": 4, "seed": 9173},
]


def _gen_inputs(B: int, D: int, S: int, W: int, seed: int):
    """NumPy stand-in for the per-shape RNG; same shapes as the torch ref."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, D, S), dtype=np.float32)
    weight = rng.standard_normal((D, W), dtype=np.float32)
    bias = rng.standard_normal((D,), dtype=np.float32)
    return x, weight, bias


def causal_conv1d_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Depthwise causal 1D conv in pure NumPy.

    x:      [B, D, S] float32
    weight: [D, W]    float32
    bias:   [D]       float32
    returns [B, D, S] float32
    """
    B, D, S = x.shape
    Dw, W = weight.shape
    assert D == Dw, f"channel mismatch: x has D={D}, weight has D={Dw}"
    assert bias.shape == (D,)

    # Causal: pad W-1 zeros on the left of the time axis.
    x_padded = np.pad(x, ((0, 0), (0, 0), (W - 1, 0)))   # [B, D, S + W - 1]

    out = np.broadcast_to(bias[None, :, None], (B, D, S)).astype(np.float32).copy()
    # Per-tap loop -- W iterations, each a vectorized add over [B, D, S].
    for k in range(W):
        # Slice of length S aligned to time t for tap k.
        x_slice = x_padded[:, :, k : k + S]                       # [B, D, S]
        w_slice = weight[None, :, k, None]                        # [1, D, 1]
        out += w_slice * x_slice
    return out


def _validate_against_pytorch(x, weight, bias, out_np, atol: float = 1e-4) -> None:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        print(f"--validate-against-pytorch: torch import failed ({e}); skipped")
        return

    xt = torch.from_numpy(x)
    wt = torch.from_numpy(weight).unsqueeze(1)   # [D, 1, W]
    bt = torch.from_numpy(bias)
    xp = F.pad(xt, (weight.shape[1] - 1, 0))
    out_t = F.conv1d(xp, wt, bias=bt, groups=x.shape[1]).numpy()

    diff = np.abs(out_np - out_t).max()
    print(f"validate-against-pytorch: max|diff| = {diff:.3e}  (atol={atol})")
    assert diff <= atol, f"NumPy ref disagrees with PyTorch: max|diff|={diff:.3e} > {atol}"
    print("  agreement OK")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, default=0,
                   help=f"test-shape index in [0, {len(TEST_SHAPES) - 1}]")
    p.add_argument("--validate-against-pytorch", action="store_true",
                   help="also run torch ref and assert agreement to atol=1e-4")
    args = p.parse_args()

    if not (0 <= args.shape < len(TEST_SHAPES)):
        print(f"ERROR: --shape must be in [0, {len(TEST_SHAPES) - 1}]")
        return 2

    shape = TEST_SHAPES[args.shape]
    print(f"shape[{args.shape}] = {shape}")
    x, weight, bias = _gen_inputs(**shape)
    out = causal_conv1d_numpy(x, weight, bias)

    print(f"out.shape  = {out.shape}")
    print(f"|out|_1    = {float(np.abs(out).sum()):.6e}")
    print(f"|out|_2    = {float(np.linalg.norm(out)):.6e}")
    print(f"out[0,0,:5] = {out[0, 0, :5].tolist()}")

    if args.validate_against_pytorch:
        _validate_against_pytorch(x, weight, bias, out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
