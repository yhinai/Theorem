"""
Triton (AMD backend) kernel for the gated DeltaNet inter-chunk state
recurrence on AMD MI300X (CDNA3).

Design notes:
  * One program per (batch, head) pair. The chunk loop is persistent inside
    the kernel: a single CU iterates NT chunks, eliminating per-chunk
    launch overhead.
  * Per-chunk matmul k_chunk.T @ v_gated_chunk has shape [K, BT] x [BT, V],
    which is dispatched via tl.dot() and lowers to MFMA instructions on
    CDNA3 Matrix Cores.
  * The fused-accumulate form tl.dot(a, b, acc=state) writes the result
    back into the same accumulator without an explicit RMW round-trip.
  * Exponentials use tl.exp2 (a single-instruction SIMT op on MI300X);
    we precompute LOG2E and multiply.
  * The gate scalar `exp(diff_t)` is folded into the smaller operand v
    (per-row scaling of the [BT, V] block) instead of scaling k. This is
    mathematically identical to scaling k.T but avoids touching the larger
    operand when V <= K, and matches the original optimization principle.
  * num_warps is selected per shape: small (B*H == 1) shapes get 8 warps
    to expose more in-CTA parallelism on a tiny grid; larger shapes use
    4 warps. CDNA3 wavefronts are 64 lanes -- num_warps=2 is intentionally
    avoided.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64  # BT
LOG2E = 1.4426950408889634


# Per-shape configs: (num_warps, num_stages)
# Keys are (B, T, H, K, V).
SHAPE_CONFIGS = {
    # tests
    (1, 64, 1, 64, 64):    {"num_warps": 8, "num_stages": 3},
    (2, 128, 4, 64, 64):   {"num_warps": 4, "num_stages": 3},
    (1, 256, 4, 64, 128):  {"num_warps": 4, "num_stages": 3},
    # benchmarks
    (1, 64, 1, 64, 64):    {"num_warps": 8, "num_stages": 3},  # noqa: F601
    (2, 512, 3, 64, 64):   {"num_warps": 4, "num_stages": 3},
    (2, 1024, 3, 64, 64):  {"num_warps": 4, "num_stages": 3},
}


def _pick_config(B: int, T: int, H: int, K: int, V: int):
    cfg = SHAPE_CONFIGS.get((B, T, H, K, V))
    if cfg is not None:
        return cfg
    # Fallback heuristic: small (B*H) -> 8 warps, else 4. Never 2.
    num_warps = 8 if (B * H) <= 1 else 4
    return {"num_warps": num_warps, "num_stages": 3}


@triton.jit
def chunk_fwd_h_kernel(
    k_ptr,         # [B, T, H, K] float32
    v_ptr,         # [B, T, H, V] float32
    g_ptr,         # [B, T, H]    float32
    h_ptr,         # [B, NT, H, K, V] float32
    # strides (in elements)
    sk_b, sk_t, sk_h, sk_k,
    sv_b, sv_t, sv_h, sv_v,
    sg_b, sg_t, sg_h,
    sh_b, sh_n, sh_h, sh_k, sh_v,
    # sizes
    T,
    NT,
    # constexpr tile sizes
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    LOG2E: tl.constexpr,
):
    # One program per (batch, head). Persistent chunk loop inside.
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_k = tl.arange(0, K)        # [K]
    off_v = tl.arange(0, V)        # [V]
    off_t = tl.arange(0, BT)       # [BT]

    # Running state h: [K, V], fp32 accumulator.
    state = tl.zeros((K, V), dtype=tl.float32)

    # Base pointers for this (b, h) pair.
    k_base = k_ptr + pid_b * sk_b + pid_h * sk_h
    v_base = v_ptr + pid_b * sv_b + pid_h * sv_h
    g_base = g_ptr + pid_b * sg_b + pid_h * sg_h
    h_base = h_ptr + pid_b * sh_b + pid_h * sh_h

    for c in range(0, NT):
        t0 = c * BT
        t_idx = t0 + off_t  # [BT]

        # k_chunk: [BT, K]
        k_chunk = tl.load(
            k_base + t_idx[:, None] * sk_t + off_k[None, :] * sk_k
        )
        # v_chunk: [BT, V]
        v_chunk = tl.load(
            v_base + t_idx[:, None] * sv_t + off_v[None, :] * sv_v
        )
        # g_chunk: [BT]
        g_chunk = tl.load(g_base + t_idx * sg_t)

        # g_end is the last element of g_chunk in this chunk.
        g_end = tl.load(g_base + (t0 + BT - 1) * sg_t)
        diff = g_end - g_chunk  # [BT]

        # exp(diff) via exp2: exp(x) = exp2(x * log2e). One MI300X SIMT op.
        scale = tl.exp2(diff * LOG2E)            # [BT]
        decay_chunk = tl.exp2(g_end * LOG2E)     # scalar

        # Fold the per-row gate into the smaller operand (v_gated).
        v_gated = v_chunk * scale[:, None]       # [BT, V]

        # Decay the running state, then accumulate k.T @ v_gated via MFMA.
        state = state * decay_chunk

        # k_chunk is [BT, K]; we need k_chunk.T as [K, BT] for the dot.
        k_t = tl.trans(k_chunk)                  # [K, BT]
        state = tl.dot(k_t, v_gated, acc=state)  # [K, V]

        # Write h[b, c, h, :, :] = state.
        h_off = (
            h_base
            + c * sh_n
            + off_k[:, None] * sh_k
            + off_v[None, :] * sh_v
        )
        tl.store(h_off, state)


def custom_kernel(data) -> torch.Tensor:
    """
    Launch the Triton kernel and return h: [B, NT, H, K, V] float32.
    """
    k: torch.Tensor = data["k"]
    v: torch.Tensor = data["v"]
    g: torch.Tensor = data["g"]
    B: int = data["B"]
    T: int = data["T"]
    H: int = data["H"]
    K: int = data["K"]
    V: int = data["V"]

    assert T % CHUNK_SIZE == 0, f"T={T} must be divisible by {CHUNK_SIZE}"
    NT = T // CHUNK_SIZE

    # Ensure contiguous on the same device. The kernel expects float32.
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    assert k.dtype == torch.float32
    assert v.dtype == torch.float32
    assert g.dtype == torch.float32

    h_out = torch.empty(B, NT, H, K, V, device=k.device, dtype=torch.float32)

    sk_b, sk_t, sk_h, sk_k = k.stride()
    sv_b, sv_t, sv_h, sv_v = v.stride()
    sg_b, sg_t, sg_h = g.stride()
    sh_b, sh_n, sh_h, sh_k, sh_v = h_out.stride()

    cfg = _pick_config(B, T, H, K, V)

    grid = (B, H)
    chunk_fwd_h_kernel[grid](
        k, v, g, h_out,
        sk_b, sk_t, sk_h, sk_k,
        sv_b, sv_t, sv_h, sv_v,
        sg_b, sg_t, sg_h,
        sh_b, sh_n, sh_h, sh_k, sh_v,
        T,
        NT,
        K=K, V=V, BT=CHUNK_SIZE,
        LOG2E=LOG2E,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    return h_out
