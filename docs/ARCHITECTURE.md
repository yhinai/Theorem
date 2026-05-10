# Architecture

This document describes the high-level shape of the Theorem GPU kernel
repository, the AMD Instinct MI300X target hardware, the software stack
versions, and how the four Triton kernels compose into the gated DeltaNet
forward pass and Mamba-2-style causal convolutions.

---

## 1. Project shape

The repository is organized around four self-contained Triton kernel modules,
a shared evaluation harness, and a small set of orchestration scripts.

```
Theorem/
├── kernels/
│   ├── causal_conv1d/        # depthwise 1D causal convolution (W=4 default)
│   │   ├── kernel.py         # Triton kernel + autotune configs
│   │   ├── reference.py      # PyTorch reference for correctness
│   │   ├── eval.py           # bench + correctness driver
│   │   ├── shapes.py         # canonical shape sweeps
│   │   └── README.md
│   ├── chunk_fwd_h/          # gated DeltaNet inter-chunk state recurrence
│   │   ├── kernel.py
│   │   ├── reference.py
│   │   ├── eval.py
│   │   ├── shapes.py
│   │   └── README.md
│   ├── chunk_fwd_o/          # gated DeltaNet chunkwise output (4 dots/block)
│   │   ├── kernel.py
│   │   ├── reference.py
│   │   ├── eval.py
│   │   └── shapes.py
│   └── recompute_w_u/        # gated DeltaNet WY-transform recompute (2 GEMMs)
│       ├── kernel.py
│       ├── reference.py
│       └── eval.py
├── docs/
│   ├── ARCHITECTURE.md       # (this file)
│   ├── OPTIMIZATIONS.md
│   └── BENCHMARKS.md
├── scripts/
│   ├── monitor_gpu.sh        # rocm-smi --csv tail
│   ├── run_sweep.py          # cross-kernel shape sweep driver
│   └── flush_l2.py           # 256 MB scratch buffer L2 flush helper
├── results/                  # CSV bench output, one file per kernel
├── utils.py                  # shared timing / L2 flush / correctness helpers
├── eval.py                   # top-level entrypoint: `python eval.py both kernels/<name>/`
├── pyproject.toml
├── requirements.txt
└── README.md
```

The split is deliberate: each kernel module is a small Triton-only unit with
its own reference, autotune configs, and shape list. The shared `utils.py`
holds the L2-flush + event-timing protocol so all four kernels report
comparable numbers.

---

## 2. Hardware target — AMD Instinct MI300X (CDNA3)

Theorem targets a single MI300X OAM module. All kernels are tuned exclusively
for this device; no other GPU family is supported.

| Property                          | Value                                          |
|-----------------------------------|------------------------------------------------|
| Architecture                      | CDNA3 (gfx942)                                 |
| Compute units (CUs)               | 304 across 8 XCDs (38 CUs/XCD)                 |
| Wavefront size                    | 64 lanes                                       |
| VGPRs / CU (32-bit)               | 512 (effectively 256 wide-VGPRs)               |
| AGPRs / CU                        | 512                                            |
| LDS / CU                          | 64 KiB                                         |
| L1 vector cache / CU              | 32 KiB                                         |
| L2 cache / XCD                    | 4 MiB (8 XCDs → 32 MiB aggregate)              |
| Infinity Cache (MALL)             | 256 MiB shared across XCDs                     |
| HBM3e capacity                    | 192 GB                                         |
| HBM3e peak bandwidth              | ~5.3 TB/s                                      |
| Peak fp16/bf16 MFMA throughput    | ~1.3 PFLOPS (dense)                            |
| Peak fp8 MFMA throughput          | ~2.6 PFLOPS (dense)                            |
| Peak fp32 MFMA throughput         | ~163 TFLOPS                                    |
| Supported MFMA tile shapes (used) | 16×16×16 fp16/bf16, 32×32×8 fp32, fp8 variants |
| Default boost clock               | ~2100 MHz                                      |

The kernel tuning in this repo makes load-bearing use of:

- The 64-lane wavefront (autotune `num_warps` is always a multiple of the
  per-CU wavefront budget; values of 4 and 8 are common, never 2).
- The 4 MiB per-XCD L2 (the persistent + L2-grouped kernels in
  `recompute_w_u` are sized so a chunk's reused operands stay resident).
- The MFMA Matrix Cores at the 16×16×16 fp16/bf16 and 32×32×8 fp32 tile
  shapes that Triton's AMD backend lowers `tl.dot` onto.

---

## 3. Software stack

| Component         | Version pin / minimum                             |
|-------------------|---------------------------------------------------|
| ROCm runtime      | 6.2 minimum, 7.x recommended                      |
| HIP runtime       | bundled with ROCm                                 |
| PyTorch           | 2.4+ ROCm 6.2 wheels (or 2.5+ ROCm 7.x wheels)    |
| Triton            | 3.1+ with AMD backend (gfx942 LLVM target)        |
| NumPy             | 1.26+                                             |
| Python            | 3.10+                                             |
| `rocm-smi`        | bundled with ROCm                                 |
| OS                | Ubuntu 22.04 LTS (other ROCm-supported distros OK)|

The Triton AMD backend lowers `tl.dot` to MFMA instructions automatically
when the operand dtype is fp16, bf16, or fp8 and the tile shape matches
one of the supported MFMA shapes. All four kernels in this repo rely on
that lowering — no inline assembly, no HIP fallback path.

---

## 4. Per-kernel data flow

Shapes are written as `[D₁, D₂, ...]`. Notation: `B` batch, `T` time/seq,
`H` heads, `D` head dim, `C` chunk size, `W` conv width.

### 4.1 causal_conv1d

A depthwise causal 1D convolution with a small fixed width `W` (default 4).
Memory-bound on MI300X; the kernel processes a large contiguous `S` block
at a time so the input window is reused across `W` output positions.

```
   X  [B, T, D]      W [D, W]      bias [D]
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              causal_conv1d (Triton)
              tile: BS × BD over (T, D)
              W is tl.constexpr (compile-time unroll)
                       ▼
                   Y [B, T, D]
```

### 4.2 chunk_fwd_h (gated DeltaNet inter-chunk state recurrence)

Computes the per-chunk hidden-state recurrence `H_{c+1} = G_c · H_c + W_cᵀ U_c`
across the chunk axis, where `G_c` is the per-chunk gate. Dot-heavy; runs
on Matrix Cores via MFMA.

```
   k [B, H, T, D_k]   v [B, H, T, D_v]   g [B, H, T]   w [B, H, T, D_k]
        │                  │                 │              │
        └──────────────────┼─────────────────┴──────────────┘
                           ▼
                  chunk_fwd_h (Triton)
                  persistent over chunks C
                  tl.dot acc=acc, MFMA fp16/bf16
                  exp2 gate, num_stages=3
                           ▼
                    h [B, H, NC, D_k, D_v]   (per-chunk states)
```

### 4.3 chunk_fwd_o (gated DeltaNet chunkwise output)

Computes the per-chunk output `O_c` from `(q, k, v, h, g)`. Single-pass:
the intra-chunk `qk` term and the global-state `q · h_c` term are fused in
one kernel — no HBM intermediate. 4 dot products per output block.

```
   q [B,H,T,D_k]  k [B,H,T,D_k]  v [B,H,T,D_v]  h [B,H,NC,D_k,D_v]  g [B,H,T]
        │              │              │                 │                │
        └──────────────┴──────────────┼─────────────────┴────────────────┘
                                      ▼
                            chunk_fwd_o (Triton)
                            single-pass: 4 × tl.dot/block
                            64×64 MFMA tiles, fp32 accumulator
                            causal mask via tl.where
                                      ▼
                                o [B, H, T, D_v]
```

### 4.4 recompute_w_u (gated DeltaNet WY-transform recompute)

Recomputes the `W` and `U` operands of the WY transform. The original
formulation is an O(C²) elementwise loop; this kernel reformulates it as
**two GEMMs per chunk**, and that single matmul rewrite is the largest
single optimization in the repo.

```
   k [B, H, T, D_k]   v [B, H, T, D_v]   beta [B, H, T]   A_inv [B, H, NC, C, C]
        │                   │                  │                    │
        └───────────────────┴──────────────────┴────────────────────┘
                                         ▼
                              recompute_w_u (Triton)
                              persistent-blocked, L2-grouped
                              GEMM-1: (k·β) → W  via tl.dot
                              GEMM-2: (v·β) → U  via tl.dot
                                         ▼
                              w [B,H,T,D_k]   u [B,H,T,D_v]
```

---

## 5. How the three DeltaNet kernels compose

Per the gated DeltaNet forward pass (Yang et al., *Gated Delta Networks*,
arXiv:2412.06464), the chunkwise forward decomposes into three steps that
correspond exactly to the three DeltaNet kernels in this repo. They run in
sequence, once per chunk-axis pass; `recompute_w_u` rebuilds the WY operands
that `chunk_fwd_h` and `chunk_fwd_o` consume.

```
     ┌───────────────────────────────────────────────────────────────┐
     │                  Gated DeltaNet forward (chunked)             │
     └───────────────────────────────────────────────────────────────┘

    inputs: q, k, v, beta, g, A_inv     (per chunk of size C)
                            │
                            ▼
            ┌───────────────────────────────┐
            │   recompute_w_u   (Step 1)    │   2 × GEMM per chunk
            │   produces W, U from k,v,β,A  │
            └──────────────┬────────────────┘
                           │  W, U
                           ▼
            ┌───────────────────────────────┐
            │   chunk_fwd_h     (Step 2)    │   inter-chunk recurrence
            │   H_{c+1} = G_c·H_c + Wᵀ·U    │   dot-heavy, MFMA
            └──────────────┬────────────────┘
                           │  H per chunk
                           ▼
            ┌───────────────────────────────┐
            │   chunk_fwd_o     (Step 3)    │   chunkwise output
            │   O_c from q, k, v, H, g      │   4 × tl.dot per block
            └──────────────┬────────────────┘
                           │
                           ▼
                       O (final output)
```

`causal_conv1d` is independent of the DeltaNet trio — it sits earlier in the
network, applied to the projected `(q, k, v)` stream in Mamba-2-style and
DeltaNet-style architectures alike. It is included in this repo because it
is the dominant memory-bound kernel in those architectures and benefits from
the same MI300X-specific tuning vocabulary (large `S` blocks, compile-time
`W` unroll, wavefront-aware `num_warps`).

---

## 6. Why these kernels matter

- **Gated DeltaNet** (arXiv:2412.06464) is a recently-published linear-attention
  variant that adds a learned gate to the DeltaNet update rule. Its chunkwise
  forward replaces the quadratic-in-T softmax attention with three linear-in-T
  primitives (the WY recompute, the inter-chunk recurrence, and the chunkwise
  output). Making each of those primitives fast on MI300X is what makes
  long-context gated DeltaNet training practical on a single device.
- **causal_conv1d** is the depthwise short-window convolution at the heart
  of Mamba-2-style state-space architectures, where it acts as a local
  mixer in front of the SSM/SSD operator. It is memory-bound (small fixed
  `W`, full sweep over `T`), so the optimization vocabulary is different
  from the DeltaNet trio: input reuse across the `W` window, large `S`
  blocks, no MFMA. It is the canonical memory-bound counterpart to the
  three compute-bound DeltaNet kernels and rounds out the workload mix
  this repo is tuned for.

The four kernels together cover the two hot spots of modern linear-time
sequence models on MI300X: the memory-bound short-conv front-end and the
compute-bound MFMA-heavy chunkwise recurrence/output back-end.

---

## 7. References

- AMD, *AMD Instinct MI300X Platform Architecture Whitepaper*,
  https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- AMD, *CDNA3 ISA Reference Guide* (gfx942), AMD developer docs.
- AMD ROCm documentation, https://rocm.docs.amd.com
- Triton AMD backend overview,
  https://triton-lang.org/main/programming-guide/chapter-3/amdgpu.html
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with
  Delta Rule*. arXiv:2412.06464 (2024).
- Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality* (Mamba-2).
  arXiv:2405.21060 (2024).
