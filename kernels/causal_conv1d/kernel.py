"""Optimized depthwise causal 1D convolution for AMD Instinct MI300X (CDNA3).

Targets ROCm 7.x via the Triton AMD backend. Memory-bound kernel; the
optimization story is:

  * Large S tiles (256/512) amortize launch overhead and exploit input
    reuse -- with W=4, adjacent output positions share 75% of their input
    window.
  * The contiguous axis (S) is the inner loop, so global loads coalesce
    naturally into 64-lane wavefronts on CDNA3.
  * The W loop is unrolled at compile time (W is a `tl.constexpr` and W <= 8),
    so the per-tap multiply-add stream becomes a fixed-length sequence the
    AMD backend can schedule against the Matrix Cores' surrounding ALU
    pipeline.
  * `num_warps in {4, 8}` -- never 2 -- so the wavefront count per program
    keeps the MI300X CUs occupied (4 warps x 64 lanes = 256 threads, the
    sweet spot for non-LDS-heavy memory-bound kernels on CDNA3).
  * `num_stages=2` enables LDS-side pipelining of the upcoming x-tile load
    over the current-tile compute.
  * Per-shape `SHAPE_CONFIGS` -- small problems get smaller tiles to expose
    more parallelism across the MI300X's 304 CUs; large problems get larger
    tiles to minimize launch and addressing overhead.
  * The grid orders the (S-tile, D-tile, batch) axes so adjacent program
    ids stay within the same channel column, which keeps reused weight
    rows hot in L2 (4 MB on MI300X).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl


__all__ = ["custom_kernel"]


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def causal_conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    D,
    S,
    stride_x_b,
    stride_x_d,
    stride_x_s,
    stride_w_d,
    stride_w_k,
    stride_o_b,
    stride_o_d,
    stride_o_s,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
):
    """Compute one [BLOCK_D, BLOCK_S] output tile for one batch element.

    Grid layout: axis-0 over S tiles, axis-1 over D tiles, axis-2 over B.
    Putting S on axis-0 keeps adjacent program ids on the same channel slab,
    so the weight row read at offset `d_offsets * stride_w_d + k` is reused
    across S tiles via the L2 cache.
    """
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Output tile coordinates.
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)        # [BLOCK_S]
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)        # [BLOCK_D]

    s_mask = s_offsets < S                                     # [BLOCK_S]
    d_mask = d_offsets < D                                     # [BLOCK_D]
    out_mask = d_mask[:, None] & s_mask[None, :]               # [BLOCK_D, BLOCK_S]

    # Bias is per-channel; broadcast across the S axis.
    bias_vals = tl.load(b_ptr + d_offsets, mask=d_mask, other=0.0)  # [BLOCK_D]
    acc = tl.broadcast_to(bias_vals[:, None], (BLOCK_D, BLOCK_S)).to(tl.float32)

    # Base addresses for x and out for this batch element.
    x_batch_ptr = x_ptr + pid_b * stride_x_b
    out_batch_ptr = out_ptr + pid_b * stride_o_b

    # Compile-time-unrolled W loop -- W <= 8, so this expands to a fixed
    # sequence of MAC operations the AMD backend can schedule densely.
    for k in tl.static_range(0, W):
        # Causal index: out position t reads input at t - (W - 1) + k.
        x_s = s_offsets - (W - 1) + k                           # [BLOCK_S]
        x_in_bounds = (x_s >= 0) & (x_s < S)
        x_load_mask = d_mask[:, None] & x_in_bounds[None, :]    # [BLOCK_D, BLOCK_S]

        x_addr = (
            x_batch_ptr
            + d_offsets[:, None] * stride_x_d
            + x_s[None, :] * stride_x_s
        )
        x_val = tl.load(x_addr, mask=x_load_mask, other=0.0)    # [BLOCK_D, BLOCK_S]

        w_addr = w_ptr + d_offsets * stride_w_d + k * stride_w_k
        w_val = tl.load(w_addr, mask=d_mask, other=0.0)         # [BLOCK_D]

        acc += x_val.to(tl.float32) * w_val[:, None].to(tl.float32)

    out_addr = (
        out_batch_ptr
        + d_offsets[:, None] * stride_o_d
        + s_offsets[None, :] * stride_o_s
    )
    tl.store(out_addr, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Per-shape launch configs
# ---------------------------------------------------------------------------
#
# Keys are the (B, D, S, W) tuples enumerated in `task.yml` (tests + benchmarks).
# Values are MI300X-tuned launch parameters:
#
#   * Small shapes -> smaller tiles, num_warps=4. More programs across the 304
#     CUs keeps the device fed when total work is tiny.
#   * Large shapes -> larger tiles, num_warps=8. Fewer programs each doing more
#     work amortizes the per-program addressing setup.
#   * num_stages=2 universally -- enough to overlap LDS loads with compute
#     without blowing the LDS budget on CDNA3.
#
# Anything not in this dict falls through to the heuristic in `_pick_config`.

SHAPE_CONFIGS: Dict[Tuple[int, int, int, int], dict] = {
    # Tests (small).
    (1, 64, 64, 4):    {"BLOCK_S": 64,  "BLOCK_D": 32, "num_warps": 4, "num_stages": 2},
    (2, 128, 128, 4):  {"BLOCK_S": 64,  "BLOCK_D": 32, "num_warps": 4, "num_stages": 2},
    (1, 256, 256, 3):  {"BLOCK_S": 128, "BLOCK_D": 32, "num_warps": 4, "num_stages": 2},
    (1, 128, 64, 8):   {"BLOCK_S": 64,  "BLOCK_D": 32, "num_warps": 4, "num_stages": 2},
    (4, 64, 128, 4):   {"BLOCK_S": 64,  "BLOCK_D": 32, "num_warps": 4, "num_stages": 2},
    # Benchmarks (large) -- autotuned on MI300X via continuous hill-climbing
    # (results/autotune_continuous_summary.csv). Insight: this kernel is
    # memory-bound; thin tiles along the channel axis (BLOCK_D=16) keep
    # programs small enough to expose all 304 CUs, while BLOCK_S = 128 with
    # num_warps=16 amortizes more output per program for the smaller shapes.
    (1, 1536, 2048, 4): {"BLOCK_S": 128, "BLOCK_D": 16, "num_warps": 16, "num_stages": 1},
    (1, 2560, 2048, 4): {"BLOCK_S": 128, "BLOCK_D": 16, "num_warps": 16, "num_stages": 1},
    (1, 2560, 4096, 4): {"BLOCK_S":  64, "BLOCK_D": 16, "num_warps":  8, "num_stages": 3},
}


def _pick_config(B: int, D: int, S: int, W: int) -> dict:
    """Look up an exact config or fall back to a size-bucketed heuristic."""
    cfg = SHAPE_CONFIGS.get((B, D, S, W))
    if cfg is not None:
        return cfg
    # Heuristic: large iff S*D >= 256k (matches the benchmark tier).
    if S * D >= 256 * 1024:
        return {"BLOCK_S": 256, "BLOCK_D": 64, "num_warps": 8, "num_stages": 2}
    return {"BLOCK_S": 64, "BLOCK_D": 32, "num_warps": 4, "num_stages": 2}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def custom_kernel(data: tuple) -> torch.Tensor:
    """Run the optimized depthwise causal 1D conv on MI300X.

    Args:
        data: (x, weight, bias) where
            x:      [B, D, S] float32
            weight: [D, W]    float32
            bias:   [D]       float32
        all on the AMD device (PyTorch ROCm exposes it as `cuda:0`).

    Returns:
        out: [B, D, S] float32 on the same device.
    """
    x, weight, bias = data
    assert x.is_contiguous(), "x must be contiguous (B, D, S)"
    assert weight.is_contiguous(), "weight must be contiguous (D, W)"
    assert bias.is_contiguous(), "bias must be contiguous (D,)"
    assert x.dtype == torch.float32 == weight.dtype == bias.dtype

    B, D, S = x.shape
    Dw, W = weight.shape
    assert D == Dw
    assert bias.shape == (D,)

    out = torch.empty_like(x)

    cfg = _pick_config(B, D, S, W)
    BLOCK_S = cfg["BLOCK_S"]
    BLOCK_D = cfg["BLOCK_D"]
    num_warps = cfg["num_warps"]
    num_stages = cfg["num_stages"]

    grid = (
        triton.cdiv(S, BLOCK_S),
        triton.cdiv(D, BLOCK_D),
        B,
    )

    causal_conv1d_kernel[grid](
        x,
        weight,
        bias,
        out,
        B,
        D,
        S,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        W=W,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
