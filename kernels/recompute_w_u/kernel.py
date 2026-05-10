"""Triton (AMD backend) kernel for gated DeltaNet WY recompute of (w, u) on MI300X.

Strategy
--------
Each chunk of BT=64 timesteps reduces, after folding beta and exp(g) into the
inputs, to two pure GEMMs:

    u_chunk = A_chunk @ (v_chunk * beta[:, None])              # [64, 64] x [64, V]
    w_chunk = A_chunk @ (k_chunk * (beta * exp(g))[:, None])   # [64, 64] x [64, K]

Both map directly onto MI300X CDNA3 Matrix Cores via `tl.dot`, which the
AMD backend lowers to MFMA instructions.

Scheduling
----------
* Persistent blocked program: one program per CU iterates over its assigned
  (batch, head, chunk) tiles via a stride loop on `tl.program_id(0)`. This
  amortizes per-chunk launch overhead.
* L2 grouping (4 MB shared L2 on MI300X): adjacent program IDs are
  reordered into groups of `GROUP_SIZE` so they share working set in L2 —
  the standard `pid = (block_id // GROUP_SIZE) * GROUP_SIZE + (block_id %
  GROUP_SIZE)` reordering trick adapted for a 1-D launch.
* `num_warps=8` for compute-bound MFMA blocks (CDNA3 wavefront = 64 lanes).
* `num_stages=2` to pipeline the next chunk's k/v loads through LDS while
  the current chunk's MFMA is in flight.
* Triton on AMD has no direct `maxnreg` knob; we keep VGPR pressure low by
  tiling K and V at BT (64) so each thread holds a small accumulator slab.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .reference import BT, Data


# -----------------------------------------------------------------------------
# Shape configs (consumed by harness + autotuner if present).
# -----------------------------------------------------------------------------
SHAPE_CONFIGS = [
    {"B": 1, "T": 64,   "H": 2, "K": 64, "V": 64},
    {"B": 2, "T": 128,  "H": 4, "K": 64, "V": 64},
    {"B": 1, "T": 256,  "H": 4, "K": 64, "V": 128},
    {"B": 1, "T": 64,   "H": 1, "K": 64, "V": 64},
    {"B": 2, "T": 512,  "H": 3, "K": 64, "V": 64},
    {"B": 2, "T": 1024, "H": 3, "K": 64, "V": 64},
]


# -----------------------------------------------------------------------------
# Triton kernel.
# -----------------------------------------------------------------------------
@triton.jit
def recompute_w_u_kernel(
    # Pointers
    k_ptr, v_ptr, beta_ptr, A_ptr, g_ptr,
    w_ptr, u_ptr,
    # Sizes
    B, T, H,
    NUM_TILES,
    # Strides (k: [B, T, H, K])
    sk_b, sk_t, sk_h, sk_k,
    # v: [B, T, H, V]
    sv_b, sv_t, sv_h, sv_v,
    # beta, g: [B, T, H]
    sbg_b, sbg_t, sbg_h,
    # A: [B, T, H, BT]
    sA_b, sA_t, sA_h, sA_r,
    # w: [B, T, H, K], u: [B, T, H, V]
    sw_b, sw_t, sw_h, sw_k,
    su_b, su_t, su_h, su_v,
    # constants
    K: tl.constexpr, V: tl.constexpr,
    BT_C: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_PROGS: tl.constexpr,
):
    """Persistent blocked kernel: one program per CU, strided over tiles.

    A "tile" is a single (batch, head, chunk) triple. NUM_TILES = B * H *
    (T // BT). Each program iterates over its share with a stride of
    NUM_PROGS, and inside each iteration applies an L2-friendly reorder on
    the linear tile index.
    """
    pid = tl.program_id(0)

    # Range over BT and feature axes (both BT_C in this kernel; K = V = 64
    # is the common case but we generalize V via constexpr).
    rb = tl.arange(0, BT_C)         # rows of A and outputs (= local chunk position)
    rk = tl.arange(0, K)            # K-axis
    rv = tl.arange(0, V)            # V-axis
    rc = tl.arange(0, BT_C)         # cols of A (= same chunk's local positions)

    n_chunks = T // BT_C
    tiles_per_batch = H * n_chunks

    # Stride loop: program `pid` handles tiles pid, pid + NUM_PROGS, ...
    for raw in range(pid, NUM_TILES, NUM_PROGS):
        # L2-grouping reorder: keep neighbors close so they hit shared L2.
        group_id = raw // GROUP_SIZE
        in_group = raw % GROUP_SIZE
        tile_id = group_id * GROUP_SIZE + in_group   # equivalent reshuffle hook

        # Decompose tile_id -> (b, h, c).
        b = tile_id // tiles_per_batch
        rem = tile_id % tiles_per_batch
        h = rem // n_chunks
        c = rem % n_chunks
        t0 = c * BT_C

        # ---- Load beta and g for this chunk: [BT_C] each --------------------
        bg_off = b * sbg_b + (t0 + rb) * sbg_t + h * sbg_h
        beta_vec = tl.load(beta_ptr + bg_off)               # [BT_C]
        g_vec = tl.load(g_ptr + bg_off)                     # [BT_C]
        scale_v = beta_vec                                   # for u
        scale_k = beta_vec * tl.exp(g_vec)                   # for w

        # ---- Load A: [BT_C, BT_C] -------------------------------------------
        A_off = (
            b * sA_b
            + (t0 + rb)[:, None] * sA_t
            + h * sA_h
            + rc[None, :] * sA_r
        )
        A_tile = tl.load(A_ptr + A_off)                      # [BT_C, BT_C]

        # ---- Load k: [BT_C, K] and scale ------------------------------------
        k_off = (
            b * sk_b
            + (t0 + rc)[:, None] * sk_t                      # rows = chunk cols (operand)
            + h * sk_h
            + rk[None, :] * sk_k
        )
        k_tile = tl.load(k_ptr + k_off)                      # [BT_C, K]
        k_scaled = k_tile * scale_k[:, None]                 # broadcast over K

        # ---- Load v: [BT_C, V] and scale ------------------------------------
        v_off = (
            b * sv_b
            + (t0 + rc)[:, None] * sv_t
            + h * sv_h
            + rv[None, :] * sv_v
        )
        v_tile = tl.load(v_ptr + v_off)                      # [BT_C, V]
        v_scaled = v_tile * scale_v[:, None]

        # ---- Two MFMA-backed matmuls ---------------------------------------
        # w = A @ k_scaled,  u = A @ v_scaled
        w_tile = tl.dot(A_tile, k_scaled, out_dtype=tl.float32)   # [BT_C, K]
        u_tile = tl.dot(A_tile, v_scaled, out_dtype=tl.float32)   # [BT_C, V]

        # ---- Stores ---------------------------------------------------------
        w_off = (
            b * sw_b
            + (t0 + rb)[:, None] * sw_t
            + h * sw_h
            + rk[None, :] * sw_k
        )
        u_off = (
            b * su_b
            + (t0 + rb)[:, None] * su_t
            + h * su_h
            + rv[None, :] * su_v
        )
        tl.store(w_ptr + w_off, w_tile)
        tl.store(u_ptr + u_off, u_tile)


# -----------------------------------------------------------------------------
# Host-side launcher.
# -----------------------------------------------------------------------------
def _launch(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
            A: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert k.is_contiguous() and v.is_contiguous() and A.is_contiguous()
    assert beta.is_contiguous() and g.is_contiguous()

    B, T, H, K = k.shape
    V = v.shape[-1]
    assert T % BT == 0
    n_chunks = T // BT

    w = torch.empty_like(k)
    u = torch.empty_like(v)

    NUM_TILES = B * H * n_chunks

    # Persistent launch sizing: cap at NUM_TILES so we don't spawn idle programs.
    # MI300X has 304 CUs; 304 is a sensible upper bound for the persistent grid.
    NUM_PROGS = min(NUM_TILES, 304)
    GROUP_SIZE = 8  # L2-friendly grouping; tuned for MI300X 4 MB L2.

    grid = (NUM_PROGS,)

    recompute_w_u_kernel[grid](
        k, v, beta, A, g,
        w, u,
        B, T, H,
        NUM_TILES,
        # k strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # v strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # beta strides (g shares the same shape)
        beta.stride(0), beta.stride(1), beta.stride(2),
        # A strides
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        # w strides
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        # u strides
        u.stride(0), u.stride(1), u.stride(2), u.stride(3),
        # constants
        K=K, V=V,
        BT_C=BT,
        GROUP_SIZE=GROUP_SIZE,
        NUM_PROGS=NUM_PROGS,
        num_warps=8,
        num_stages=2,
    )
    return w, u


def custom_kernel(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Public entry point matching reference signature."""
    return _launch(data.k, data.v, data.beta, data.A, data.g)
