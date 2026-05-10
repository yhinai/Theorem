# Optimizations

This document describes the per-kernel optimization techniques applied across
the four Triton kernels in this repository, expressed in MI300X / CDNA3 / ROCm
terms. The vocabulary is consistent across kernels: tile sizing, wavefront
sizing, MFMA tile selection on Matrix Cores, software pipelining via
`num_stages`, persistent kernels, L2 grouping for the 4 MiB per-XCD L2, and
compile-time specialization through `tl.constexpr`.

The four kernels span the two regimes that matter on MI300X:

- **Memory-bound** (HBM-throughput-limited): `causal_conv1d`.
- **Compute-bound** (MFMA-throughput-limited): `chunk_fwd_h`, `chunk_fwd_o`,
  `recompute_w_u`.

The optimization vocabulary differs accordingly. Memory-bound kernels
maximize bytes-per-launch and exploit operand reuse across the small fixed
window. Compute-bound kernels maximize MFMA issue rate, hide HBM latency
under compute via `num_stages` software pipelining, and keep reused operands
resident in the per-XCD L2.

---

## Causal Conv1D

`causal_conv1d` is a depthwise causal 1D convolution with a small fixed
kernel width `W` (default 4) over the time axis. For each `(b, t, d)` output
it reads `W` consecutive input elements along `t` and a `W`-element weight
vector for that channel `d`. It is memory-bound on MI300X because the
arithmetic intensity is tiny — `W` multiply-adds per loaded input — so HBM
read bandwidth is the dominant constraint and the design problem is to make
sure each loaded byte is reused as many times as the access pattern allows.

### Optimization techniques applied

- **Large `S` blocks along the time axis.** Each program instance processes
  a contiguous block of `BS` time positions per channel block (typical
  `BS = 64..256`). With `W = 4` and a sliding window, every loaded input
  element is reused by up to `W` output positions, giving a 75% input
  overlap; large `BS` amortizes the program-launch and address-arithmetic
  overhead across that reuse. The bigger the `S` block, the closer the
  kernel runs to the HBM bandwidth wall.
- **Coalesced `S` access.** Loads along the contiguous `T` axis are issued
  as wavefront-wide vector loads so 64 lanes co-issue 64 consecutive
  fp16/bf16 elements per cycle, saturating the per-CU L1 vector-cache port
  and minimizing duplicate HBM traffic. The kernel iterates `D` (the
  depthwise channel axis) on the outer loop and `S` on the inner loop so
  that the inner-loop strides are unit-stride in HBM.
- **Compile-time `W` unroll.** `W` is passed to the kernel as
  `tl.constexpr`, so the Triton AMD backend fully unrolls the `W`-element
  inner reduction. This eliminates the loop control flow, replaces the
  weight-vector load with `W` scalar literals after constant folding, and
  lets the scheduler pack the `W` FMAs as a contiguous dependency chain in
  VGPRs.
- **Per-shape autotune configs.** The shape sweep enumerates `(BS, BD,
  num_warps, num_stages)` configs separately for each canonical
  `(B, T, D)` shape, because the optimal block size depends on whether `D`
  is small enough to fit a wide `BD` (large channels-per-block, fewer
  programs, better wavefront occupancy) or whether `T` is short enough that
  `BS` saturates instead.
- **Wavefront-aware `num_warps` (4–8, never 2).** With a 64-lane wavefront,
  `num_warps = 2` leaves a CU at 128 lanes — too narrow to hide HBM latency
  on a memory-bound kernel. `num_warps = 4` (256 lanes) is the floor;
  `num_warps = 8` (512 lanes) is preferred for the wider-`BD` configs where
  the extra wavefronts let the CU keep more outstanding HBM requests in
  flight under the memory subsystem's MSHR limit.
- **No MFMA path.** The arithmetic shape (`W = 4` reductions) is far too
  small to amortize an MFMA tile, so the kernel uses scalar fp16/bf16 FMAs
  in VGPRs. This is the right choice on a memory-bound kernel: MFMA would
  add register pressure without raising the HBM-bandwidth ceiling.

---

## Chunk Forward H (`chunk_fwd_h`)

`chunk_fwd_h` computes the gated DeltaNet inter-chunk hidden-state recurrence
`H_{c+1} = G_c · H_c + W_cᵀ · U_c` along the chunk axis, where `G_c` is the
per-chunk gate folded from the per-token gate `g`. It is dot-heavy: each
chunk step is two matrix products in `(D_k, D_v)`. The kernel runs persistent
over the chunk axis so the running state `H_c` stays live in registers
between chunks instead of round-tripping through HBM.

### Optimization techniques applied

- **MFMA matmul on Matrix Cores.** Both `tl.dot` calls in the kernel are
  emitted at 16×16×16 fp16/bf16 tile shapes, which the Triton AMD backend
  lowers to MFMA `v_mfma_f32_16x16x16_f16` / `..._bf16` instructions. With
  the fp32 accumulator chosen, the per-CU peak is ~660 GFLOPS at boost,
  multiplying out across 304 CUs to the ~1.3 PFLOPS dense fp16/bf16 figure.
- **`exp2` over `exp`.** The per-token gate `g` is reduced to a per-chunk
  scalar via `exp2(log2_e * sum(g))`. `exp2` lowers to a single hardware
  transcendental on CDNA3 with no software-emulated mantissa step;
  `exp(x) = exp2(x * 1.4426950408889634)` is strictly slower because the
  pre-scale adds an FMA and pushes the result through a wider polynomial.
- **Gate-diff folded into the smaller operand.** The chunkwise gate
  difference is multiplied into the smaller of the two GEMM operands
  (typically the `D_k`-side `W` operand rather than the `D_v`-side `U`),
  so the elementwise scaling happens on `O(C·D_k)` data rather than
  `O(C·D_v)`. On the demo shapes `D_k = 64..128` while `D_v = 128..256`,
  giving 2× to 4× fewer scaling FMAs without changing the final
  accumulator.
- **Fused dot-accumulate via `tl.dot(acc=...)`.** The recurrence
  `H_{c+1} = G_c · H_c + W_cᵀ · U_c` is expressed as a single
  `tl.dot(a, b, acc=acc)` call where `acc` carries the gated previous
  state. This emits a continuous chain of MFMA instructions sharing the
  fp32 accumulator register file with no intervening writeback, which on
  CDNA3 keeps the AGPR-based accumulator resident across chunk steps.
- **`num_stages = 3` inner pipelining.** The chunk-axis loop is software-
  pipelined with three stages: prefetch-next-operands, MFMA on current,
  writeback (folded into the next iteration's `acc`). Three stages is the
  sweet spot for MI300X here — two stages leaves the MFMA pipe waiting on
  HBM for the next operand pair, four stages spills VGPRs and triggers
  AGPR-to-VGPR moves that cost more than they save.
- **Persistent loop over chunks.** Each program instance owns a fixed
  `(b, h, D_k_block, D_v_block)` tile and walks all `NC` chunks for that
  tile in a single launch. The running `H` state lives in registers
  between chunks; the alternative — relaunching per chunk — would write
  `H` to HBM every step, which on long sequences dominates the runtime.
- **Per-shape `num_warps`.** Configs sweep `num_warps ∈ {4, 8}`. The
  smaller `D_k = 64` shapes prefer 4 warps (less VGPR pressure, more
  programs per CU); the wider `D_k = 128` shapes prefer 8 warps because
  the wider MFMA tile geometry covers more output per program and the
  extra wavefronts hide the operand-prefetch latency.

---

## Chunk Forward O (`chunk_fwd_o`)

`chunk_fwd_o` is the chunkwise output kernel: given `(q, k, v, h, g)` it
produces the per-token output `O` for each chunk. It is the most
compute-bound of the four kernels — four matrix products per output block:
the intra-chunk `q · kᵀ` (gated, causally masked), the intra-chunk
`(qkᵀ) · v`, the global-state `q · h_c`, and the global-state combination
`(q · h_c) · v_aux` term. Crucially, the kernel is **single-pass**: the
intra-chunk `qk` term and the global-state term are fused in one kernel
with no HBM intermediate.

### Optimization techniques applied

- **Single-pass structure (no HBM intermediate).** A naive implementation
  computes the intra-chunk `qkᵀ`, writes it to HBM, then loads it back to
  multiply by `v`. This kernel keeps the intermediate `qkᵀ` block in
  registers and immediately consumes it with the next `tl.dot`. On MI300X
  this saves an `O(B·H·T·C)` round-trip — at the demo shapes that's tens
  of GB per forward pass. The savings are the largest single contributor
  to the kernel's measured speedup over the chained reference.
- **MFMA fp32 accumulator on 64×64 tiles.** Each `tl.dot` is emitted at a
  64×64 output tile with an fp32 accumulator, which the Triton AMD backend
  composes from 16×16×16 MFMA tiles (4×4 grid per block). The fp32
  accumulator avoids the precision loss that would compound across four
  chained dots; the 64×64 tile is wide enough to amortize the operand-load
  cost over the four MFMAs that share each `q`/`k`/`v`/`h` operand.
- **`exp2` gate.** Same rationale as `chunk_fwd_h`: the per-token gate is
  reduced via `exp2`, which is one hardware transcendental on CDNA3.
- **Causal masking via `tl.where`.** The intra-chunk causal mask is applied
  with `tl.where(row >= col, qk, -inf)` after the first `tl.dot` and
  before the softmax-style scaling. `tl.where` on CDNA3 lowers to
  predicated VGPR moves; the `-inf` lane is then squashed by the
  subsequent exp without a branch. This keeps the entire mask in the
  vector pipeline — no scalar control flow per tile.
- **`num_warps = 8` or `16` for compute-bound 4-dot blocks.** Four MFMA
  dots per output block put a lot of accumulator pressure on the AGPR
  file; widening `num_warps` to 8 (default) or 16 (for the large-`D_v`
  shapes) gives more wavefronts to round-robin through the MFMA pipe so
  the long dependency chain through the four dots stays full. `num_warps
  = 4` underutilizes the MFMA pipe on this kernel — the chain is long
  enough that two wavefronts can't keep it busy.
- **Persistent across the chunk axis.** Like `chunk_fwd_h`, each program
  instance owns a fixed output tile and walks all chunks for that tile in
  one launch, so `q` and `h` operands are loaded once and reused across
  chunks where the access pattern allows.

---

## Recompute W,U (`recompute_w_u`)

`recompute_w_u` rebuilds the `W` and `U` operands of the gated DeltaNet
WY-transform from `(k, v, beta, A_inv)`. The original numpy/PyTorch
formulation is an O(C²) elementwise loop over the within-chunk axis. This
kernel reformulates that loop as **two GEMMs per chunk** — `(k · β) · A_inv
→ W` and `(v · β) · A_inv → U` — collapsing the elementwise loop into MFMA
work. That single matmul rewrite is the largest single optimization in the
repo: it changes the kernel's complexity class from O(C²) elementwise
arithmetic to O(C²) MFMA arithmetic, and MFMA on MI300X runs at roughly
two orders of magnitude higher throughput than scalar fp16/bf16 FMAs.

### Optimization techniques applied

- **Matmul reformulation (the largest single optimization).** The within-
  chunk recurrence that builds `W` and `U` is rewritten as two
  `tl.dot(k_block · beta, A_inv)` and `tl.dot(v_block · beta, A_inv)`
  GEMMs. The C² elementwise loop becomes one MFMA per output tile. On
  MI300X this is the difference between scalar VGPR FMAs running at the
  per-CU vector throughput and Matrix Core MFMAs running at the dense
  fp16/bf16 peak — a roughly 50–100× per-CU arithmetic-throughput swing
  on the inner work.
- **Persistent-blocked kernel.** The kernel is persistent over the
  `(B, H)` grid and blocked along the chunk axis, with each program
  instance owning a fixed `(b, h, chunk_block)` tile. `A_inv` for that
  chunk is loaded once into registers and reused across both the `W` and
  `U` GEMMs, halving the HBM traffic on the `(C, C)` operand.
- **L2 grouping (4 MiB cache reordering).** Program IDs are reordered with
  an L2-locality grouping so that adjacent programs on the grid share an
  XCD's 4 MiB L2. With `D_k`/`D_v` operand blocks sized to keep two
  consecutive chunks' `k`/`v` slices co-resident, the second program in
  a group hits L2 instead of HBM on the shared operand. This is the same
  pattern as the canonical L2-aware MFMA matmul launch grid.
- **`num_warps = 8+`.** With two back-to-back MFMA dots per chunk and the
  `A_inv` operand staying resident across both, the kernel benefits from
  more wavefronts to keep the MFMA pipe busy through the operand swap.
  `num_warps = 8` is the default; `num_warps = 16` helps for the largest
  `(D_k, D_v)` configs where the per-program tile is wide enough to
  justify the extra wavefronts.
- **`num_stages = 2`.** Two-stage software pipelining is enough here
  because the chunk-axis trip count is small and the operand reuse on
  `A_inv` already covers most of the latency that `num_stages = 3` would
  hide. Going to three stages costs VGPRs and pushes the kernel into
  AGPR-spill territory on the wider configs without measurably improving
  MFMA issue rate.
- **`tl.constexpr` shape specialization.** `BLOCK_C`, `BLOCK_DK`, and
  `BLOCK_DV` are `tl.constexpr` so the chunk-block tile shape is baked
  into the compiled kernel. The compiler unrolls the inner MFMA grid and
  resolves all stride arithmetic at compile time.

---

## Cross-cutting patterns

The four kernels share a small vocabulary of techniques that recur with
slight variation. The table below summarizes the cross-cutting patterns and
which kernels apply each.

| Pattern                          | causal_conv1d | chunk_fwd_h | chunk_fwd_o | recompute_w_u |
|----------------------------------|:---:|:---:|:---:|:---:|
| Static shapes (no dynamic dims)  | yes | yes | yes | yes |
| `tl.constexpr` block sizes       | yes | yes | yes | yes |
| `tl.constexpr` `W` / chunk size  | yes | yes | yes | yes |
| Persistent kernel over outer axis| no  | yes | yes | yes |
| MFMA tile sizing on Matrix Cores | no  | 16×16×16 | 16×16×16 (64×64 block) | 16×16×16 |
| fp32 MFMA accumulator            | n/a | yes | yes | yes |
| `exp2` over `exp`                | n/a | yes | yes | n/a |
| `num_stages = 2`                 | sometimes | no | sometimes | yes |
| `num_stages = 3`                 | sometimes | yes | yes | no |
| Wavefront-aware `num_warps`      | 4–8 | 4–8 | 8–16 | 8–16 |
| L2 grouping (4 MiB / XCD)        | no  | implicit | implicit | yes (explicit) |
| Per-shape autotune configs       | yes | yes | yes | yes |
| Coalesced contiguous-axis loads  | yes | yes | yes | yes |

### Notes on the recurring vocabulary

- **Static shapes + `tl.constexpr`.** Every problem dimension that
  participates in the inner loop is `tl.constexpr`. This collapses
  address arithmetic to immediates, lets the compiler fully unroll inner
  loops, and lets the autotuner pick a different specialized binary per
  shape rather than carrying a dynamic loop bound at runtime.
- **Persistent kernels.** For all three DeltaNet kernels, the chunk axis
  is walked inside the kernel rather than across launches. This keeps
  reused state (`H` in `chunk_fwd_h`, `q`/`h` in `chunk_fwd_o`, `A_inv`
  in `recompute_w_u`) live in registers across chunks and avoids HBM
  round-trips on data that is reused within a few microseconds.
- **MFMA tile sizing.** All MFMA work uses the 16×16×16 fp16/bf16 shape
  with an fp32 accumulator. `chunk_fwd_o` composes a 64×64 output tile
  out of a 4×4 grid of those 16×16×16 MFMAs; `chunk_fwd_h` and
  `recompute_w_u` use narrower output tiles. The fp32 accumulator is
  load-bearing for numerical accuracy across chained dots.
- **Wavefront sizing principles.** `num_warps` is always a multiple of 1
  whole wavefront (64 lanes) and never 2. The lower bound is 4 (256
  lanes) for memory-bound work, 8 (512 lanes) for compute-bound work,
  and 16 (1024 lanes) for the widest `chunk_fwd_o` and `recompute_w_u`
  configs where the MFMA dependency chain is longest.
- **Per-shape autotune.** Each kernel ships an autotune config space
  enumerated by `(block sizes, num_warps, num_stages)` and a canonical
  shape list in `kernels/<name>/shapes.py`. The autotuner picks the best
  config per shape; the picked configs are committed so the demo path
  doesn't pay autotune cost on first launch.

---

## References

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
