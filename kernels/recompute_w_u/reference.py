"""Pure PyTorch reference for gated DeltaNet WY-transform recompute of (w, u).

Generates (k, v, beta, A, g) deterministically from a seed. The matrix A is
itself produced by the upstream WY-transform stage, which we replicate here
so this kernel can be tested standalone:

    For each chunk of length BT:
        K_c    = k[chunk]                              # [BT, K]
        beta_c = beta[chunk]                           # [BT]
        g_c    = g[chunk]                              # [BT]   (cumulative, chunk-local)
        M      = (K_c @ K_c.T) * beta_c[:, None]       # [BT, BT]
                 * exp(g_c[:, None] - g_c[None, :])    # gated decay
        M      = tril(M, diagonal=-1)                  # strictly lower
        A      = (I - M)^{-1}                          # solve via tri solve
        # store A[chunk] in [BT, BT] block

The (w, u) recompute is then:

    u = A @ (v * beta[:, None])
    w = A @ (k * (beta * exp(g))[:, None])

Both are pure [64, 64] x [64, K] / [64, 64] x [64, V] matmuls per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BT = 64  # chunk length, fixed by the upstream WY transform


@dataclass
class Data:
    k: torch.Tensor       # [B, T, H, K]
    v: torch.Tensor       # [B, T, H, V]
    beta: torch.Tensor    # [B, T, H]
    A: torch.Tensor       # [B, T, H, BT]
    g: torch.Tensor       # [B, T, H]
    B: int
    T: int
    H: int
    K: int
    V: int


def _build_A(k: torch.Tensor, beta: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Replicates the upstream WY-transform that produces A.

    k:    [B, T, H, K]
    beta: [B, T, H]
    g:    [B, T, H]   (chunk-local cumulative gate)

    Returns A: [B, T, H, BT] — for each timestep t belonging to chunk c,
    A[..., t, :] is the row of A_c that corresponds to t's local position in c.
    """
    B, T, H, K = k.shape
    assert T % BT == 0, "T must be a multiple of BT (=64)"
    n_chunks = T // BT

    # Reshape to per-chunk: [B, n_chunks, BT, H, K]
    k_c = k.reshape(B, n_chunks, BT, H, K)
    beta_c = beta.reshape(B, n_chunks, BT, H)
    g_c = g.reshape(B, n_chunks, BT, H)

    # Move H next to the chunk dim for batched matmul: [B, n_chunks, H, BT, K]
    k_c = k_c.permute(0, 1, 3, 2, 4).contiguous()
    beta_c = beta_c.permute(0, 1, 3, 2).contiguous()
    g_c = g_c.permute(0, 1, 3, 2).contiguous()

    # KK^T: [B, n_chunks, H, BT, BT]
    KKT = torch.matmul(k_c, k_c.transpose(-1, -2))

    # Gated decay factor exp(g_i - g_j)
    gate = torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))   # [B, n_chunks, H, BT, BT]

    M = KKT * beta_c.unsqueeze(-1) * gate                      # [..., BT, BT]
    # Strictly lower (zero diagonal and upper)
    mask = torch.tril(torch.ones(BT, BT, device=k.device, dtype=torch.bool), diagonal=-1)
    M = M * mask

    eye = torch.eye(BT, device=k.device, dtype=k.dtype).expand_as(M)
    # A = (I - M)^{-1}; (I - M) is unit lower triangular, so a triangular solve is exact.
    A_chunks = torch.linalg.solve_triangular(eye - M, eye, upper=False, unitriangular=True)
    # A_chunks: [B, n_chunks, H, BT, BT]

    # Lay back out into [B, T, H, BT]: for each chunk c, row r of A_c lives at t = c*BT + r.
    A_chunks = A_chunks.permute(0, 1, 3, 2, 4).contiguous()    # [B, n_chunks, BT, H, BT]
    A = A_chunks.reshape(B, T, H, BT)
    return A


def generate_data(B: int, T: int, H: int, K: int, V: int, seed: int,
                  device: str = "cuda", dtype: torch.dtype = torch.float32) -> Data:
    """Deterministic data generation for tests/benchmarks."""
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)

    k = torch.randn(B, T, H, K, generator=g_cpu, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, generator=g_cpu, dtype=dtype) * 0.1
    # beta in (0, 1)
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g_cpu, dtype=dtype))
    # g: small chunk-local cumulative gate; build per-chunk cumsum of small negatives.
    raw_g = -torch.rand(B, T, H, generator=g_cpu, dtype=dtype) * 0.05
    assert T % BT == 0
    raw_g = raw_g.reshape(B, T // BT, BT, H).cumsum(dim=2).reshape(B, T, H)

    k = k.to(device)
    v = v.to(device)
    beta = beta.to(device)
    raw_g = raw_g.to(device)

    A = _build_A(k, beta, raw_g)

    return Data(k=k, v=v, beta=beta, A=A, g=raw_g, B=B, T=T, H=H, K=K, V=V)


def ref_kernel(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference (w, u) recompute.

    u = A @ (v * beta[:, None])
    w = A @ (k * (beta * exp(g))[:, None])
    """
    k, v, beta, A, g = data.k, data.v, data.beta, data.A, data.g
    B, T, H, K = k.shape
    V = v.shape[-1]
    assert T % BT == 0
    n_chunks = T // BT

    # Per-chunk reshape: [B, n_chunks, BT, H, *]
    k_c = k.reshape(B, n_chunks, BT, H, K).permute(0, 1, 3, 2, 4).contiguous()
    v_c = v.reshape(B, n_chunks, BT, H, V).permute(0, 1, 3, 2, 4).contiguous()
    beta_c = beta.reshape(B, n_chunks, BT, H).permute(0, 1, 3, 2).contiguous()
    g_c = g.reshape(B, n_chunks, BT, H).permute(0, 1, 3, 2).contiguous()

    # Reconstruct A as [B, n_chunks, H, BT, BT] from the [B, T, H, BT] storage.
    A_c = A.reshape(B, n_chunks, BT, H, BT).permute(0, 1, 3, 2, 4).contiguous()

    # Fold beta / (beta * exp(g)) into v / k.
    v_scaled = v_c * beta_c.unsqueeze(-1)
    k_scaled = k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1)

    u_c = torch.matmul(A_c, v_scaled)   # [B, n_chunks, H, BT, V]
    w_c = torch.matmul(A_c, k_scaled)   # [B, n_chunks, H, BT, K]

    # Lay back out to [B, T, H, *]
    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).contiguous()
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).contiguous()
    return w, u


# Harness alias
generate_input = generate_data
