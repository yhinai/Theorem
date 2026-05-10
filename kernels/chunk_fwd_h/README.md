# chunk_fwd_h — Gated DeltaNet inter-chunk state recurrence (AMD MI300X)

Forward pass for the **inter-chunk state** of a gated DeltaNet-style linear
attention layer, written in Triton (AMD backend) for **AMD MI300X
(CDNA3)** under ROCm.

> This kernel computes the per-chunk **state** `h[B, NT, H, K, V]` only.
> The intra-chunk attention output (per-token `o`) is the job of the
> sibling kernel `chunk_fwd_o` and is **not** computed here.

---

## Math

The sequence of length `T` is partitioned into non-overlapping chunks of
`BT = 64` timesteps. For each chunk `c` and each `(batch, head)` pair we
maintain a running state matrix `h[K, V]`:

```
g_chunk[t]   = cumulative gate within the chunk (precomputed upstream in g)
g_end        = g_chunk[BT - 1]
diff_t       = g_end - g_chunk[t]
v_gated[t]   = v[t] * exp(diff_t)             # elementwise on V
h_new        = h_old * exp(g_end) + k_chunk.T @ v_gated_chunk
```

`h_new` has shape `[K, V]`. The matmul is `[K, BT] x [BT, V]` with
`BT = 64` and (in the test set) `K = 64`, `V in {64, 128}`.

The output is written as `h[b, c, hd, :, :] = h_new` for each chunk index
`c in [0, NT)` where `NT = T // BT`.

---

## Inputs / Output

| Tensor | Shape | dtype | Description |
| --- | --- | --- | --- |
| `k` | `[B, T, H, K]` | `float32` | keys |
| `v` | `[B, T, H, V]` | `float32` | values |
| `g` | `[B, T, H]` | `float32` | cumulative gate, already cumsum'd within each chunk upstream |
| `h` (out) | `[B, NT, H, K, V]` | `float32` | per-chunk inter-state |

`T` must be a multiple of `BT = 64`.

---

## Shapes covered

```yaml
tests:
  - {B: 1, T: 64,   H: 1, K: 64, V: 64}
  - {B: 2, T: 128,  H: 4, K: 64, V: 64}
  - {B: 1, T: 256,  H: 4, K: 64, V: 128}
benchmarks:
  - {B: 1, T: 64,   H: 1, K: 64, V: 64}
  - {B: 2, T: 512,  H: 3, K: 64, V: 64}
  - {B: 2, T: 1024, H: 3, K: 64, V: 64}
tolerances: {rtol: 1e-2, atol: 1e-2}
```

---

## Optimization rationale (CDNA3-specific)

1. **Matrix-engine dots.** The per-chunk `k_chunk.T @ v_gated_chunk`
   is `[64, 64] x [64, V]`. Triton's `tl.dot()` lowers to **MFMA**
   instructions on CDNA3 Matrix Cores — the right primitive for this
   tile shape.

2. **`exp2` instead of `exp`.** On MI300X, `tl.exp2(x)` is a single SIMT
   instruction. We use the identity `exp(x) = exp2(x * log2e)` for both
   the per-row gate `exp(diff_t)` and the chunk-end decay `exp(g_end)`.

3. **Fold the gate into the smaller operand.** The mathematically
   equivalent identity
   `k.T @ (diag(s) @ v) = (k.T) @ (s[:, None] * v)`
   lets us scale the `[BT, V]` block instead of the `[BT, K]` block.
   When `V <= K` this is strictly cheaper; the principle is preserved
   uniformly across the shape set.

4. **Fused dot-accumulate.** `tl.dot(k_t, v_gated, acc=state)`
   accumulates into the running `state` register tile without an
   explicit read-modify-write to LDS/registers between the decay step
   and the matmul.

5. **Inner-loop pipelining.** `num_stages = 3` lets the compiler
   prefetch the next chunk's `k`, `v`, and `g` while the current chunk's
   MFMA is in flight.

6. **Per-shape `num_warps`.** Selected from `SHAPE_CONFIGS`:

   | Shape | `num_warps` | Why |
   | --- | --- | --- |
   | `B=1, H=1` (small grid)         | **8** | Tiny `(B, H)` grid (1 CTA) — push more wavefront-level parallelism inside the CTA. |
   | All other test/benchmark shapes | **4** | Larger `B*H` grid amortizes occupancy across CUs; 4 warps balances register pressure for `[K, V] = [64, 64..128]`. |

   CDNA3 wavefronts are 64 lanes; `num_warps=2` is intentionally avoided.

7. **Persistent kernel.** The grid is `(B, H)` and the chunk loop runs
   **inside** the kernel. One program per `(batch, head)` pair iterates
   all `NT` chunks, so launch overhead is paid once per pair instead of
   once per chunk. The state register tile lives across the whole chunk
   loop.

---

## Files

```
chunk_fwd_h/
├── task.yml      # shapes, tolerances, description
├── reference.py  # eager PyTorch reference (slow but correct)
├── kernel.py     # Triton (AMD backend) kernel + custom_kernel(data)
├── __init__.py   # exports custom_kernel, ref_kernel
└── README.md     # this file
```

---

## How to run

```python
import torch
from kernels.chunk_fwd_h import custom_kernel, ref_kernel
from kernels.chunk_fwd_h.reference import generate_input

data = generate_input(B=2, T=512, H=3, K=64, V=64, seed=4052)

# Move tensors to the MI300X device under ROCm before launching:
device = torch.device(0)            # ROCm-visible MI300X (device 0)
for key in ("k", "v", "g"):
    data[key] = data[key].to(device)

h_ref  = ref_kernel(data)
h_fast = custom_kernel(data)
torch.testing.assert_close(h_fast, h_ref, rtol=1e-2, atol=1e-2)
```

The MI300X is exposed via PyTorch's ROCm build; the device handle above
addresses it as device index `0`.

---

## Scope: this is the inter-chunk recurrence only

This kernel produces the **state** `h[B, NT, H, K, V]` — the
right-hand context that each chunk hands off to the next. The
**intra-chunk** attention output (the per-token `o[B, T, H, V]` that a
DeltaNet layer actually returns) is computed by the sibling kernel
`chunk_fwd_o`, which consumes this `h` together with the same `(k, v, g)`
and the chunk's queries `q`. Splitting the two passes is what lets each
one specialize its tiling and use its own MFMA-friendly shape.
