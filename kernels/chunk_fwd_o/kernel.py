"""
Triton (AMD backend) kernel for gated DeltaNet chunkwise output (forward, o).

Target: AMD MI300X (CDNA3, ROCm). Tile shapes are sized for MFMA Matrix Cores
(32x32x8 or 16x16x16 fp32 paths). All four dot products in the chunkwise math
are fused in a single kernel pass:

    1) qk        = q_chunk @ k_chunk.T          (BT x K) @ (K x BT)  -> (BT x BT)
    2) local_out = causal_mask(qk) @ v_gated    (BT x BT) @ (BT x V) -> (BT x V)
    3) global    = q_gated @ h[chunk_idx]       (BT x K) @ (K x V)   -> (BT x V)
    4) o_chunk   = scale * (local_out + global)

The intermediate [BT, BT] qk tile is kept in registers / LDS - never spilled
to HBM. We use `tl.exp2(x * log2e)` for the gate's hardware fast-path on
CDNA3 and apply the causal mask via `tl.where` after the qk dot.

Per-shape configs are keyed by (B, T, H, K, V) so we can pick num_warps /
num_stages tuned for the smallest vs largest benchmark shapes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Per-shape Triton (AMD backend) launch configs.
#
# CDNA3 wavefronts are 64 lanes; for compute-bound 4-dot blocks with BT=64 we
# benefit from more warps (num_warps>=8) than the 2-4 typical of smaller
# matmul kernels. num_stages=2 enables LDS pipelining of the q/k/v/h loads.
# ---------------------------------------------------------------------------
SHAPE_CONFIGS: Dict[Tuple[int, int, int, int, int], Dict[str, int]] = {
    # Tests
    (1, 64, 1, 64, 64):    {"BT": 64, "BV": 64,  "num_warps": 8,  "num_stages": 2},
    (2, 128, 4, 64, 64):   {"BT": 64, "BV": 64,  "num_warps": 8,  "num_stages": 2},
    (1, 256, 4, 64, 128):  {"BT": 64, "BV": 64,  "num_warps": 8,  "num_stages": 2},
    # Benchmarks
    (1, 64,   1, 64, 64):  {"BT": 64, "BV": 64,  "num_warps": 8,  "num_stages": 2},
    (2, 512,  3, 64, 64):  {"BT": 64, "BV": 64,  "num_warps": 16, "num_stages": 2},
    (2, 1024, 3, 64, 64):  {"BT": 64, "BV": 64,  "num_warps": 16, "num_stages": 2},
}

DEFAULT_CONFIG: Dict[str, int] = {"BT": 64, "BV": 64, "num_warps": 8, "num_stages": 2}


@triton.jit
def _chunk_fwd_o_kernel(
    Q_ptr, K_ptr, V_ptr, G_ptr, H_ptr, O_ptr,
    scale,
    # strides
    sq_b, sq_t, sq_h, sq_k,
    sk_b, sk_t, sk_h, sk_k,
    sv_b, sv_t, sv_h, sv_v,
    sg_b, sg_t, sg_h,
    sh_b, sh_n, sh_h, sh_k, sh_v,
    so_b, so_t, so_h, so_v,
    # sizes
    T, H, K, V, NT,
    # tile / meta
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Grid: (B*H, NT, ceil(V/BV))
        pid_bh -> batch * head
        pid_nt -> chunk index along T
        pid_v  -> output-feature tile along V
    """
    pid_bh = tl.program_id(0)
    pid_nt = tl.program_id(1)
    pid_v  = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh %  H

    LOG2E = 1.4426950408889634

    # Row indices inside the chunk and value-feature indices for this V tile.
    offs_t = tl.arange(0, BT)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    mask_v = offs_v < V

    t0 = pid_nt * BT

    # ----- load q_chunk [BT, BK] and gate g_chunk [BT] -----
    q_row_off = (b * sq_b) + ((t0 + offs_t)[:, None] * sq_t) + (h * sq_h) + (offs_k[None, :] * sq_k)
    q_chunk = tl.load(Q_ptr + q_row_off)                                   # [BT, BK]

    g_off = (b * sg_b) + ((t0 + offs_t) * sg_t) + (h * sg_h)
    g_chunk = tl.load(G_ptr + g_off)                                       # [BT]

    # q_gated = q * exp(g) ; v_gated = v * exp(-g)
    exp_g     = tl.exp2(g_chunk * LOG2E)
    exp_neg_g = tl.exp2((-g_chunk) * LOG2E)

    q_gated = q_chunk * exp_g[:, None]                                     # [BT, BK]

    # ----- load k_chunk [BT, BK] -----
    k_row_off = (b * sk_b) + ((t0 + offs_t)[:, None] * sk_t) + (h * sk_h) + (offs_k[None, :] * sk_k)
    k_chunk = tl.load(K_ptr + k_row_off)                                   # [BT, BK]

    # ----- load v_chunk [BT, BV] and apply gate-diff scaling -----
    v_row_off = (b * sv_b) + ((t0 + offs_t)[:, None] * sv_t) + (h * sv_h) + (offs_v[None, :] * sv_v)
    v_mask = mask_v[None, :]
    v_chunk = tl.load(V_ptr + v_row_off, mask=v_mask, other=0.0)           # [BT, BV]
    v_gated = v_chunk * exp_neg_g[:, None]                                 # [BT, BV]

    # ----- 1) qk = q_gated @ k_chunk.T  -> [BT, BT] ; causal mask -----
    # tl.dot uses MFMA Matrix Cores on CDNA3; fp32 accumulator.
    qk = tl.dot(q_gated, tl.trans(k_chunk), allow_tf32=False)              # [BT, BT]
    causal = offs_t[None, :] <= offs_t[:, None]                            # [BT, BT]
    qk = tl.where(causal, qk, 0.0)

    # ----- 2) local_out = qk @ v_gated  -> [BT, BV] -----
    local_out = tl.dot(qk, v_gated, allow_tf32=False)                      # [BT, BV]

    # ----- 3) global = q_gated @ h[b, pid_nt, h, :, :]  -> [BT, BV] -----
    h_row_off = (b * sh_b) + (pid_nt * sh_n) + (h * sh_h) + \
                (offs_k[:, None] * sh_k) + (offs_v[None, :] * sh_v)
    h_mask = mask_v[None, :]
    h_tile = tl.load(H_ptr + h_row_off, mask=h_mask, other=0.0)            # [BK, BV]
    global_out = tl.dot(q_gated, h_tile, allow_tf32=False)                 # [BT, BV]

    # ----- 4) o_chunk = scale * (local_out + global_out) -----
    o_chunk = scale * (local_out + global_out)

    o_row_off = (b * so_b) + ((t0 + offs_t)[:, None] * so_t) + (h * so_h) + (offs_v[None, :] * so_v)
    tl.store(O_ptr + o_row_off, o_chunk, mask=mask_v[None, :])


def _select_config(B: int, T: int, H: int, K: int, V: int) -> Dict[str, int]:
    cfg = SHAPE_CONFIGS.get((B, T, H, K, V))
    if cfg is None:
        cfg = DEFAULT_CONFIG
    return cfg


def custom_kernel(data: Dict[str, Any]) -> torch.Tensor:
    """
    Entry point. Dispatches the fused chunkwise-output kernel on AMD MI300X
    via Triton (AMD backend).

    Args:
        data: dict with q, k, v, g, h, scale (and optional B/T/H/K/V).

    Returns:
        o: [B, T, H, V] float32 tensor.
    """
    q: torch.Tensor = data["q"]
    k: torch.Tensor = data["k"]
    v: torch.Tensor = data["v"]
    g: torch.Tensor = data["g"]
    h: torch.Tensor = data["h"]
    scale: float = float(data["scale"])

    assert q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and h.is_cuda, \
        "all inputs must be on a ROCm/MI300X device"
    assert q.dtype == torch.float32, "fp32 expected"

    B, T, H, K = q.shape
    Vdim = v.shape[-1]
    NT = h.shape[1]

    cfg = _select_config(B, T, H, K, Vdim)
    BT = cfg["BT"]
    BV = cfg["BV"]
    num_warps = cfg["num_warps"]
    num_stages = cfg["num_stages"]

    assert T % BT == 0, f"T={T} must be divisible by BT={BT}"
    assert NT == T // BT, f"NT={NT} must equal T/BT={T // BT}"

    BK = K  # K is small (64) and divides one MFMA group -> single-K-tile dot.

    o = torch.empty(B, T, H, Vdim, dtype=q.dtype, device=q.device)

    grid = (B * H, NT, triton.cdiv(Vdim, BV))

    _chunk_fwd_o_kernel[grid](
        q, k, v, g, h, o,
        scale,
        # q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # k strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # v strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # g strides
        g.stride(0), g.stride(1), g.stride(2),
        # h strides
        h.stride(0), h.stride(1), h.stride(2), h.stride(3), h.stride(4),
        # o strides
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # sizes
        T, H, K, Vdim, NT,
        # constexprs
        BT=BT, BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o
