"""Reference implementation for depthwise causal 1D convolution.

Pure PyTorch reference used to validate the optimized AMD MI300X (CDNA3) kernel
in `kernel.py`. Inputs are generated deterministically via a per-shape seed so
the reference and the optimized path always see byte-identical tensors.

Math:
    out[b, d, t] = bias[d] + sum_{k=0..W-1} weight[d, k] * x[b, d, t - W + 1 + k]

with out-of-bounds inputs treated as zero (causal left zero-pad of width W-1).
Each of the D channels is convolved independently (groups=D in conv1d-speak).

PyTorch's ROCm build exposes the AMD device under the `cuda:0` device-name
alias, so we use that string here -- it is the supported public API on ROCm 7.x.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


_DEVICE = "cuda:0"


def generate_input(
    B: int,
    D: int,
    S: int,
    W: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically generate (x, weight, bias) for a given shape + seed.

    Args:
        B: batch size.
        D: channel count (also the conv group count -- depthwise).
        S: sequence length.
        W: kernel width (number of taps).
        seed: per-shape RNG seed.

    Returns:
        x:      [B, D, S] float32 on the AMD device.
        weight: [D, W]    float32 on the AMD device.
        bias:   [D]       float32 on the AMD device.
    """
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    x = torch.randn((B, D, S), device=_DEVICE, dtype=torch.float32, generator=gen)
    weight = torch.randn((D, W), device=_DEVICE, dtype=torch.float32, generator=gen)
    bias = torch.randn((D,), device=_DEVICE, dtype=torch.float32, generator=gen)
    return x, weight, bias


def ref_kernel(data) -> torch.Tensor:
    """Reference depthwise causal 1D conv.

    Implemented via `F.pad` (causal left zero-pad of width W-1) and `F.conv1d`
    with `groups=D`, which matches the math above exactly.

    Args:
        data: tuple `(x, weight, bias)` returned by `generate_input`.

    Returns:
        out: [B, D, S] float32.
    """
    x, weight, bias = data
    B, D, S = x.shape
    Dw, W = weight.shape
    assert D == Dw, f"channel mismatch: x has D={D}, weight has D={Dw}"
    assert bias.shape == (D,), f"bias shape {tuple(bias.shape)} != ({D},)"

    # Causal: pad W-1 zeros on the left, none on the right.
    x_padded = F.pad(x, (W - 1, 0))

    # Depthwise conv1d expects weight of shape [D, 1, W] when groups=D.
    w = weight.unsqueeze(1)
    out = F.conv1d(x_padded, w, bias=bias, groups=D)

    assert out.shape == (B, D, S), f"output shape {tuple(out.shape)} != {(B, D, S)}"
    return out
