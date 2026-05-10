# chunk_fwd_o — gated DeltaNet chunkwise output (forward, o)

AMD MI300X (CDNA3, ROCm) Triton (AMD backend) kernel for the forward `o` pass of
gated DeltaNet, computed chunkwise. Pairs with `chunk_fwd_h` (which produces the
inter-chunk state `h`).

## Math

Per chunk of length `BT = 64`, with `q_gated = q * exp(g)` and
`v_gated = v * exp(-g)`:

```
o = scale * (local_attention + global_state)

local_attention[b, h, t, v] = sum_{s <= t in chunk} (q_gated[t] . k[s]) * v_gated[s]
global_state[b, h, t, v]    = q_gated[t] . h[chunk_idx]
```

Concretely, four dot products are fused per chunk:

1. `qk        = q_gated @ k.T`            `[64, K] x [K, 64]   -> [64, 64]`
2. `local_out = causal_mask(qk) @ v_gated` `[64, 64] x [64, V] -> [64, V]`
3. `global    = q_gated @ h[chunk_idx]`   `[64, K] x [K, V]    -> [64, V]`
4. `o_chunk   = scale * (local_out + global)`

The causal mask zeroes out the strict upper triangle of `qk` (so contributions
sum only over `s <= t` within the chunk).

## Inputs / Output

| Tensor | Shape           | Dtype  | Notes                                     |
| ------ | --------------- | ------ | ----------------------------------------- |
| q      | [B, T, H, K]    | fp32   | queries                                   |
| k      | [B, T, H, K]    | fp32   | keys                                      |
| v      | [B, T, H, V]    | fp32   | values                                    |
| g      | [B, T, H]       | fp32   | cumulative gate (small magnitude)         |
| h      | [B, NT, H, K, V]| fp32   | inter-chunk state from `chunk_fwd_h`      |
| scale  | scalar          | fp32   | typically `1/sqrt(K)`                     |
| o      | [B, T, H, V]    | fp32   | output                                    |

`NT = T / BT` with `BT = 64`.

## Optimization rationale (MI300X / CDNA3)

- **Single-pass fused kernel.** All four dots execute in one launch per
  `(batch*head, chunk, V-tile)`. The intermediate `[64, 64]` `qk` matrix never
  hits HBM — it lives in registers / LDS between dots 1 and 2.
- **MFMA tile sizing.** `BT = 64`, `BK = K = 64`, `BV = 64` map cleanly onto
  MI300X Matrix Cores via `tl.dot`. The shapes match the natural CDNA3 MFMA
  groupings (32×32×8 / 16×16×16 fp32). The fp32 accumulator is forced via
  `allow_tf32=False`.
- **Hardware-fast gate.** `exp(x)` is computed as `tl.exp2(x * log2e)` so the
  CDNA3 fast-path executes the exponential.
- **Causal mask via `tl.where`.** The mask is applied after dot-1, before
  dot-2, so the lower-triangular region of `qk` propagates and the upper
  triangle is zeroed (sum form, so masking to `0` is correct here).
- **Wavefront-aware launch.** CDNA3 wavefronts are 64 lanes wide. The kernel
  is compute-bound (4 dots of mixed `K`/`V`), so we use `num_warps=8` for the
  smaller benchmark shapes and `num_warps=16` for the larger ones (`T=512`,
  `T=1024`). `num_stages=2` enables LDS pipelining of the `q/k/v/h` loads.
- **Per-shape configs.** `SHAPE_CONFIGS` in `kernel.py` is keyed by
  `(B, T, H, K, V)` so the smallest test shape and the largest benchmark
  shape can use distinct `num_warps`. `DEFAULT_CONFIG` covers any unkeyed
  shape.
- **Grid layout.** `grid = (B*H, NT, ceil(V/BV))`. Each program owns one
  chunk's worth of rows for a single `(batch, head)` and a `BV`-wide tile of
  the value dimension. With `V <= 128` and `BV = 64`, the inner V-loop is
  zero or one extra tile — minimal overhead.

## Files

- `task.yml` — shapes, seeds, tolerances, description.
- `reference.py` — pure PyTorch eager loop, used for correctness checks.
  Builds deterministic inputs via `_build_inputs(B, T, H, K, V, seed)`.
- `kernel.py` — Triton (AMD backend) kernel + `custom_kernel(data) -> o`
  entry point. `SHAPE_CONFIGS` selects per-shape launch params.
- `__init__.py` — exports `custom_kernel` and `ref_kernel`.

## Running

```python
from kernels.chunk_fwd_o import custom_kernel, ref_kernel
from kernels.chunk_fwd_o.reference import _build_inputs

data = _build_inputs(B=2, T=128, H=4, K=64, V=64, seed=5236, device="cuda")
o_ref = ref_kernel(data)
o_out = custom_kernel(data)
torch.testing.assert_close(o_out, o_ref, rtol=1e-2, atol=1e-2)
```

`device="cuda"` here is the PyTorch ROCm convention on MI300X — the underlying
runtime is ROCm and the kernel is JIT-compiled by Triton's AMD backend.

## Numerical notes

- All inputs and accumulators are fp32. `allow_tf32=False` forces fp32 MFMA.
- The gate is bounded (`g ~ N(0, 0.05^2)` in the reference inputs) so
  `exp(±g)` stays in a comfortable fp32 range across the test/benchmark
  shapes.
- Tolerances are `rtol=1e-2, atol=1e-2`, matching the upstream task.
