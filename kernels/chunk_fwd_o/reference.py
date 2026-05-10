"""
Pure PyTorch reference implementation of gated DeltaNet chunkwise output (forward, o).

This is the ground-truth eager loop used to validate the Triton (AMD backend) kernel
on AMD MI300X (CDNA3 / ROCm). Determinism is achieved per-input via the seed field
in `data`.

Math (per chunk of length BT):
    o = scale * (local_attention + global_state)

    local_attention[b, h, t, v] = sum_{s <= t in chunk} (q[t] . k[s]) * v_gated[s]
    global_state[b, h, t, v]    = q_gated[t] . h[chunk_idx]
    q_gated = q * exp(g_at_t)
    v_gated = v * exp(diff_in_chunk)

Shapes:
    q: [B, T, H, K]
    k: [B, T, H, K]
    v: [B, T, H, V]
    g: [B, T, H]
    h: [B, NT, H, K, V]
    o: [B, T, H, V]
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch


BT_DEFAULT = 64


def _build_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    seed: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Build deterministic inputs for the given shape + seed."""
    g_cpu = torch.Generator(device="cpu").manual_seed(int(seed))

    BT = BT_DEFAULT
    assert T % BT == 0, f"T={T} must be divisible by BT={BT}"
    NT = T // BT

    q = torch.randn(B, T, H, K, generator=g_cpu, dtype=dtype).to(device)
    k = torch.randn(B, T, H, K, generator=g_cpu, dtype=dtype).to(device)
    v = torch.randn(B, T, H, V, generator=g_cpu, dtype=dtype).to(device)
    # Keep gate values bounded so exp() doesn't blow up the reference.
    g = (torch.randn(B, T, H, generator=g_cpu, dtype=dtype) * 0.05).to(device)
    h = (torch.randn(B, NT, H, K, V, generator=g_cpu, dtype=dtype) * 0.1).to(device)

    scale = 1.0 / math.sqrt(K)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "h": h,
        "scale": scale,
        "B": B,
        "T": T,
        "H": H,
        "K": K,
        "V": V,
        "BT": BT,
        "NT": NT,
        "seed": seed,
    }


def ref_kernel(data: Dict[str, Any]) -> torch.Tensor:
    """
    Eager, per-chunk reference for chunkwise gated DeltaNet output.

    Returns:
        o: [B, T, H, V] float32 tensor on the same device as inputs.
    """
    q = data["q"]
    k = data["k"]
    v = data["v"]
    g = data["g"]
    h = data["h"]
    scale = float(data["scale"])

    B, T, H, K = q.shape
    _, _, _, V = v.shape
    BT = int(data.get("BT", BT_DEFAULT))
    NT = T // BT

    o = torch.zeros(B, T, H, V, dtype=q.dtype, device=q.device)

    # Loop over chunks. Inside each chunk we form the [BT, BT] qk tile, apply a
    # causal mask, then contract against gated v and add the global state hop.
    for b in range(B):
        for ht in range(H):
            for ci in range(NT):
                t0 = ci * BT
                t1 = t0 + BT

                q_chunk = q[b, t0:t1, ht, :]              # [BT, K]
                k_chunk = k[b, t0:t1, ht, :]              # [BT, K]
                v_chunk = v[b, t0:t1, ht, :]              # [BT, V]
                g_chunk = g[b, t0:t1, ht]                 # [BT]
                h_chunk = h[b, ci, ht, :, :]              # [K, V]

                # Gate-weighted q: q_gated[t] = q[t] * exp(g[t])
                q_gated = q_chunk * torch.exp(g_chunk).unsqueeze(-1)   # [BT, K]

                # Gate-diff weighted v inside the chunk:
                # v_gated[s] uses exp(g[t] - g[s]) when accumulated against k[s].
                # We absorb the per-row exp(g[t]) into q_gated above and the
                # per-col exp(-g[s]) into v_gated below, so the [BT, BT] qk matmul
                # is plain. This matches the chunkwise gated DeltaNet derivation.
                v_gated = v_chunk * torch.exp(-g_chunk).unsqueeze(-1)  # [BT, V]

                # 1) qk = q_gated @ k.T  -> [BT, BT]
                qk = q_gated @ k_chunk.transpose(0, 1)

                # Causal mask within the chunk: keep s <= t.
                idx = torch.arange(BT, device=q.device)
                causal = idx.unsqueeze(0) <= idx.unsqueeze(1)          # [BT, BT]
                qk_masked = torch.where(causal, qk, torch.zeros_like(qk))

                # 2) local_out = qk_masked @ v_gated  -> [BT, V]
                local_out = qk_masked @ v_gated

                # 3) global = q_gated @ h[chunk_idx] -> [BT, V]
                global_out = q_gated @ h_chunk

                # 4) sum + scale
                o[b, t0:t1, ht, :] = scale * (local_out + global_out)

    return o
