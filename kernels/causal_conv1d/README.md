# causal_conv1d -- Depthwise Causal 1D Convolution (AMD MI300X)

Memory-bound depthwise 1D conv with causal (left) zero-padding. Used as the
short-range mixing primitive in Mamba-2-style state-space models. Targets
**AMD Instinct MI300X (CDNA3, gfx942)** under **ROCm 7.x** via the Triton
AMD backend.

## Math

For each batch `b`, channel `d`, and time index `t in [0, S)`:

```
out[b, d, t] = bias[d] + sum_{k=0..W-1} weight[d, k] * x[b, d, t - W + 1 + k]
```

Out-of-range inputs (`t - W + 1 + k < 0`) are treated as zero -- this is the
causal left zero-pad of width `W - 1`. Channels are independent (depthwise:
`groups = D`).

## Inputs / outputs

| Name   | Shape    | Dtype   | Notes                  |
|--------|----------|---------|------------------------|
| x      | [B, D, S] | float32 | contiguous, on `cuda:0`|
| weight | [D, W]    | float32 | contiguous             |
| bias   | [D]       | float32 | contiguous             |
| out    | [B, D, S] | float32 | same device as x       |

`cuda:0` is the PyTorch ROCm device-name alias for the AMD GPU.

## Shapes

Test set (small, correctness):

| B | D   | S   | W |
|---|-----|-----|---|
| 1 | 64  | 64  | 4 |
| 2 | 128 | 128 | 4 |
| 1 | 256 | 256 | 3 |
| 1 | 128 | 64  | 8 |
| 4 | 64  | 128 | 4 |

Benchmark set (large, throughput):

| B | D    | S    | W |
|---|------|------|---|
| 1 | 1536 | 2048 | 4 |
| 1 | 2560 | 2048 | 4 |
| 1 | 2560 | 4096 | 4 |

Tolerances: `rtol = 1e-2`, `atol = 1e-2`.

## Optimization rationale (CDNA3-specific)

* **Wavefront sizing.** CDNA3 wavefronts are 64 lanes wide. We pin
  `num_warps in {4, 8}` -- never 2 -- so each program holds 256 or 512
  threads. That keeps the MI300X compute units fed without saturating LDS.
* **Large S tiles.** With `W = 4`, adjacent output positions share 75%
  of their input window. Big `BLOCK_S` (256 for the benchmark tier)
  amortizes per-tile address arithmetic and pulls each input element
  into registers once for many output reuses.
* **Coalesced loads on the S axis.** S is the contiguous (innermost)
  dimension of x. Putting `BLOCK_S` lanes along that axis means each
  load is a single coalesced transaction per wavefront.
* **Compile-time unrolled W loop.** `W` is a `tl.constexpr` (3, 4, or 8),
  so `tl.static_range(0, W)` expands at compile time. The AMD backend
  sees a fixed-length stream of multiply-add ops it can schedule against
  the surrounding load/store pipeline.
* **Causal masking via `tl.where`-style load mask.** We compute
  `x_in_bounds = (x_s >= 0) & (x_s < S)` per S lane and pass it as the
  load mask with `other=0.0`. No separate padding buffer, no branch
  divergence -- masked lanes simply contribute zero.
* **LDS pipelining.** `num_stages=2` lets the next iteration's x-tile
  load overlap the current iteration's accumulate. Two stages is enough
  on a memory-bound kernel; more stages would only inflate the LDS
  budget without freeing more bandwidth.
* **L2 reuse via grid ordering.** The launch grid is
  `(cdiv(S, BLOCK_S), cdiv(D, BLOCK_D), B)`. Ordering S on axis-0 and D
  on axis-1 keeps neighboring program ids on the same channel slab, so
  the weight rows (`[D, W]`) stay hot in MI300X's 4 MB L2 across S tiles.
* **Per-shape configs.** `SHAPE_CONFIGS` maps each task-yml entry to a
  hand-picked tile + warp count. Small shapes get
  `BLOCK_S=64, BLOCK_D=32, num_warps=4` to spread work across the 304
  CUs; large shapes get `BLOCK_S=256, BLOCK_D=64, num_warps=8` to cut
  per-program overhead. Unknown shapes fall through a size-bucket
  heuristic.

## How to run

```python
from kernels.causal_conv1d import custom_kernel, ref_kernel
from kernels.causal_conv1d.reference import generate_input

x, weight, bias = generate_input(B=1, D=2560, S=4096, W=4, seed=54352)
expected = ref_kernel(x, weight, bias)
actual = custom_kernel((x, weight, bias))
assert (actual - expected).abs().max().item() < 1e-2
```
