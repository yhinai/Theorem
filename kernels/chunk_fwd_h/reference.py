"""
Pure-PyTorch reference for the gated DeltaNet inter-chunk state recurrence.

This is the slow-but-correct definition. The Triton kernel in kernel.py is
verified against this on AMD MI300X under ROCm.
"""

from __future__ import annotations

import torch

CHUNK_SIZE = 64  # BT


def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int):
    """
    Generate (k, v, g) inputs for the inter-chunk recurrence.

    Shapes:
        k: [B, T, H, K]   float32  -- keys
        v: [B, T, H, V]   float32  -- values
        g: [B, T, H]      float32  -- per-position cumulative gate
                                     (already cumsum'd within chunks)

    The gate g is constructed so that exp(g) does not overflow / vanish for
    the chunk lengths in the test set: small negative increments per step,
    cumsum'd within each chunk independently.
    """
    assert T % CHUNK_SIZE == 0, f"T={T} must be a multiple of CHUNK_SIZE={CHUNK_SIZE}"

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    k = torch.randn(B, T, H, K, generator=gen, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, H, V, generator=gen, dtype=torch.float32) * 0.5

    # Per-position gate increment: small negative, plus a tiny noise.
    # Cumsum within each chunk so g[chunk_end] is the chunk's total decay.
    g_inc = (
        -0.05 * torch.rand(B, T, H, generator=gen, dtype=torch.float32)
        - 0.005
    )
    NT = T // CHUNK_SIZE
    g_inc = g_inc.view(B, NT, CHUNK_SIZE, H)
    g = torch.cumsum(g_inc, dim=2).reshape(B, T, H).contiguous()

    return {"k": k, "v": v, "g": g, "B": B, "T": T, "H": H, "K": K, "V": V}


def ref_kernel(data) -> torch.Tensor:
    """
    Reference per-chunk recurrence in eager PyTorch.

    Returns:
        h: [B, NT, H, K, V] float32
    """
    k = data["k"]
    v = data["v"]
    g = data["g"]
    B = data["B"]
    T = data["T"]
    H = data["H"]
    K = data["K"]
    V = data["V"]

    BT = CHUNK_SIZE
    assert T % BT == 0
    NT = T // BT

    device = k.device
    dtype = torch.float32

    h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)

    for b in range(B):
        for hd in range(H):
            state = torch.zeros(K, V, device=device, dtype=dtype)  # h_old
            for c in range(NT):
                t0 = c * BT
                t1 = t0 + BT

                k_chunk = k[b, t0:t1, hd, :]      # [BT, K]
                v_chunk = v[b, t0:t1, hd, :]      # [BT, V]
                g_chunk = g[b, t0:t1, hd]         # [BT]

                g_end = g_chunk[-1]               # scalar
                diff = g_end - g_chunk            # [BT]

                # v_gated[t] = v[t] * exp(diff_t)
                v_gated = v_chunk * torch.exp(diff).unsqueeze(-1)  # [BT, V]

                # h_new = h_old * exp(g_end) + k.T @ v_gated
                state = state * torch.exp(g_end) + k_chunk.transpose(0, 1) @ v_gated

                h_out[b, c, hd] = state

    return h_out
