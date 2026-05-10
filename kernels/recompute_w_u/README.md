# `recompute_w_u` — Gated DeltaNet WY-Transform Recompute (AMD MI300X)

One of three forward-pass kernels for **gated DeltaNet**
(paper: [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)). Rather than
persisting `w` and `u` as activations across the forward/backward boundary,
the model **recomputes** them from `(k, v, beta, A, g)` — the classic
memory-vs-compute tradeoff: trade a couple of small per-chunk GEMMs for not
having to keep two large `[B, T, H, K]` / `[B, T, H, V]` activation tensors
live in HBM.

This kernel targets **AMD MI300X (CDNA3)** via the **Triton AMD backend**
and lowers the heavy work onto **Matrix Cores (MFMA instructions)**.

---

## Math

Time is partitioned into chunks of length `BT = 64`. Per chunk
`(b, h, c)` (batch, head, chunk index):

```
u_c = A_c @ (v_c * beta_c[:, None])              # [64, 64] x [64, V]  -> [64, V]
w_c = A_c @ (k_c * (beta_c * exp(g_c))[:, None]) # [64, 64] x [64, K]  -> [64, K]
```

`A` is the **WY representation** of the chunk's gated update operator,
produced by an upstream stage from `k k^T` plus a unit-lower triangular
solve:

```
M_c = tril((K_c K_c^T) * beta_c[:, None] * exp(g_c[:, None] - g_c[None, :]),
           diagonal=-1)
A_c = (I - M_c)^{-1}
```

`reference.py` builds `A` from `(k, beta, g)` so the kernel is testable
standalone; in production it is provided by the upstream WY-transform
kernel.

Once `beta` and `exp(g)` are folded into `k` and `v`, the body of the
recompute is **two pure GEMMs per chunk** — exactly the shape MI300X's
Matrix Cores are designed for.

---

## Shapes

```
k:    [B, T, H, K]   float32
v:    [B, T, H, V]   float32
beta: [B, T, H]      float32
A:    [B, T, H, BT]  float32        (BT = 64; A_c laid out as 64 rows of 64)
g:    [B, T, H]      float32        (chunk-local cumulative gate)

w:    [B, T, H, K]   float32
u:    [B, T, H, V]   float32
```

`T` must be a multiple of `BT = 64`. Test shapes use `K, V ∈ {64, 128}`.

---

## MI300X Optimization Rationale

### 1. Matmul reformulation (the principle)
The naïve recurrence has an O(C²) elementwise inner loop per chunk. By
folding the gate and `beta` into `k`/`v`, both updates become single
GEMMs with `A_c` as the left operand. The inner loop disappears; what
remains is exactly what CDNA3 Matrix Cores are best at.

### 2. MFMA tile sizing
With `BT = K = V = 64`, each per-chunk GEMM is `[64, 64] x [64, 64]` (or
`[64, 64] x [64, 128]` in the V=128 case). `tl.dot` lowers cleanly onto
**MFMA `16x16x4` / `32x32x8` `f32`** chained tiles. We keep accumulators
in `float32` (`out_dtype=tl.float32`) since the kernel runs end-to-end in
fp32 per the upstream contract.

### 3. Persistent blocked scheduling
Total tiles = `B * H * (T // 64)`; for typical demo shapes that's between
2 and 96 — small numbers where launch overhead dominates if you spawn
one program per tile. We instead launch a **persistent** grid of
`min(NUM_TILES, 304)` programs (304 = MI300X CU count) and have each
program stride over its assigned tiles via `tl.program_id(0) +
NUM_PROGS * step`. One launch, zero per-chunk dispatch overhead.

### 4. L2 grouping (4 MB shared L2)
MI300X has a 4 MB L2 shared across CUs. We reorder the persistent
program's tile sequence into groups of `GROUP_SIZE = 8` (the standard
`(group_id * G) + (id % G)` reorder), so adjacent CUs hammer adjacent
tiles whose `A`, `k`, `v` slabs overlap in L2. Cuts redundant HBM traffic
for the `A`/`k` operands when neighboring chunks share batch/head.

### 5. `num_warps=8`, `num_stages=2`
- **`num_warps=8`** — CDNA3 wavefronts are 64 lanes; the two
  back-to-back `[64, 64] x [64, 64+]` GEMMs are compute-bound on MFMA
  and benefit from 8 warps' worth of lane parallelism, hiding load
  latency behind issue.
- **`num_stages=2`** — pipelines the *next* chunk's `k`/`v` loads
  through LDS while the current chunk's MFMA tail is still in flight.
  Going higher (3+) blew VGPR budget without improving cycle count in
  this shape regime.

### 6. VGPR / occupancy
Triton on AMD has no direct `maxnreg` knob equivalent. We instead keep
the per-thread accumulator footprint small: BT and the K/V tile sizes
are both `<= 128`, so each thread's slab of the `[64, V]` accumulator
fits comfortably in VGPRs and we maintain high CU occupancy. The
`tl.constexpr` constants on `K`, `V`, `BT_C`, `GROUP_SIZE`, and
`NUM_PROGS` let the AMD backend specialize and unroll the small inner
axes.

### 7. Loop unrolling
The chunk-dimension loop is bounded and small for typical
`(B, H, T)`; the `range(pid, NUM_TILES, NUM_PROGS)` stride loop is
unrolled by the compiler since `NUM_TILES` is constexpr-known per
launch.

---

## Files

```
recompute_w_u/
├── task.yml          # shapes + tolerances + math description
├── reference.py      # PyTorch reference + deterministic data generation
├── kernel.py         # Triton (AMD) kernel + host launcher
├── __init__.py       # package exports
└── README.md         # this file
```

---

## How to run

The harness consumes `custom_kernel(data) -> (w, u)` and `ref_kernel(data)
-> (w, u)`. Test/benchmark shapes are declared in `task.yml`. Tolerances:
`rtol = atol = 1e-2`.

```python
from kernels.recompute_w_u import generate_data, custom_kernel, ref_kernel

data = generate_data(B=2, T=128, H=4, K=64, V=64, seed=5236, device="cuda")
w_ref, u_ref = ref_kernel(data)
w_out, u_out = custom_kernel(data)
torch.testing.assert_close(w_out, w_ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(u_out, u_ref, rtol=1e-2, atol=1e-2)
```

(`device="cuda"` is the PyTorch HIP/ROCm device handle on MI300X.)
