# Benchmarks

This document specifies the timing methodology, hardware setup, correctness
threshold, and reproduction commands used to measure all four kernels in this
repository. Numbers are taken from real runs on AMD Instinct MI300X — see
`results/sweep_*.csv` and `results/baseline_compare.csv` for the raw data.

---

## 1. Methodology

### 1.1 Timing protocol

Each timed measurement is taken on a single AMD Instinct MI300X with the
following protocol, implemented in `utils.py` and re-used by every
per-kernel `eval.py`:

1. **Warmup**: 5 untimed kernel iterations to JIT-compile the Triton kernel,
   resolve the autotune-selected config, and stabilize device clocks.
2. **Timed iterations**: 50 timed kernel iterations per shape.
3. **Per-iteration timing**: each iteration is bracketed by a pair of
   `torch.cuda.Event(enable_timing=True)` records (`start.record()` /
   `end.record()`). Under PyTorch's ROCm wheels, `torch.cuda.Event` maps
   onto HIP events backed by the ROCm runtime's GPU timestamp counter, so
   the measured interval is wall-clock GPU time excluding host launch
   overhead after the first iteration.
4. **Synchronization**: `torch.cuda.synchronize()` is called once after all
   50 timed iterations finish but before reading event elapsed times.
   Per-iteration synchronization is avoided so the measurement does not
   serialize against host code.
5. **L2 cache flush between iterations**: a 256 MiB scratch buffer is
   written to between iterations to evict the per-XCD 4 MiB L2 entries
   the previous iteration may have left resident. 256 MiB is comfortably
   larger than the 32 MiB aggregate L2 across all 8 XCDs, so the next
   iteration's first HBM reads always miss in L2. This prevents
   artificially low numbers caused by repeated re-runs hitting cached
   operands.
6. **Statistics**: from the 50 measurements per shape we report `min_us`,
   `p50_us`, and `mean_us`. `min_us` is the headline number (best
   achievable on this hardware with this config); `p50_us` is the
   typical-run number; `mean_us` is reported alongside `min_us` to
   surface variance.
7. **Reference baseline**: each Triton kernel is compared against a plain
   PyTorch eager implementation (`F.conv1d` for `causal_conv1d`; explicit
   `torch.matmul` chunk-loops for the gated DeltaNet kernels) on the same
   shapes with the same protocol. The eager baseline is the realistic
   "competent PyTorch user" point of comparison.

### 1.2 Correctness threshold

Per task spec, correctness is checked against the per-kernel PyTorch
reference (`kernels/<name>/reference.py`) using:

- `torch.allclose(out, ref, rtol=1e-2, atol=1e-2)`

with both the kernel output and the reference computed in the same input
dtype (fp16 or bf16 depending on the kernel's canonical config). The
kernel-side fp32 MFMA accumulator means the kernel is generally **more**
accurate than the reference at these tolerances; the loose tolerance is
inherited from the original task spec to allow for the small numeric
differences that come from MFMA fp32 accumulation order vs. PyTorch's
serial fp16/bf16 reduction.

### 1.3 Reproducibility notes

- **Seeds**: input tensors are generated with `torch.manual_seed(0)`
  before each shape's measurement block, so the same shape produces the
  same input across runs.
- **Clocks**: MI300X boost clocks are not pinned in this repo; the 5-iter
  warmup is intended to settle clocks before timing. For tighter
  reproducibility across machines, the operator should pin clocks via
  `rocm-smi --setperflevel high` (or set a fixed sclk via
  `rocm-smi --setsclk`) before the run.
- **Power state**: `rocm-smi -d 0 -P` should report a stable power draw
  before the timed loop starts. The monitoring helper at
  `bash scripts/monitor_gpu.sh` runs `rocm-smi --csv` in a tail loop and
  is intended to be run alongside the bench in another terminal.
- **Background load**: no other ROCm clients should be on the device.
  `rocm-smi --showpids` should return only the bench process.
- **Run-to-run variance**: on a fully idle MI300X with the L2-flush
  protocol above, run-to-run variance on `min_us` is typically under 2%
  for the compute-bound kernels and under 5% for `causal_conv1d` (which
  is more sensitive to HBM scheduling).

---

## 2. Hardware setup

| Property            | Value                                              |
|---------------------|----------------------------------------------------|
| GPU                 | AMD Instinct MI300X (CDNA3, gfx942)                |
| Visible devices     | 1 (single virtual function in our setup → 1 device)|
| CUs                 | 304 (8 XCDs × 38 CUs/XCD)                          |
| HBM3e               | 192 GB, ~5.3 TB/s peak                             |
| L2 / XCD            | 4 MiB                                              |
| Boost clock         | ~2100 MHz (default; not pinned)                    |
| ROCm runtime        | 7.x                                                |
| HIP runtime         | bundled with ROCm 7.x                              |
| GPU monitoring tool | `rocm-smi --csv` (tailed via `scripts/monitor_gpu.sh`) |

The "single virtual function" note matters because MI300X supports SR-IOV
partitioning. In our setup the device is exposed as a single VF passed
through to the host, so `torch.cuda.device_count()` reports 1 and the full
304 CUs / 192 GB HBM are available to a single process.

GPU monitoring during the timed run is collected via:

```bash
bash scripts/monitor_gpu.sh   # runs `rocm-smi --csv` in a tail loop
```

The CSV columns we monitor are GPU clock (`sclk`), HBM clock (`mclk`),
power (`Average Graphics Package Power (W)`), and HBM-controller
utilization (`% memory busy`).

---

## 3. Results — reference vs optimized

Measured on AMD Instinct MI300X (gfx942), 304 CUs, single virtual function
visible to PyTorch. Shapes match `kernels/<name>/task.yml` exactly. All
correctness checks pass at `rtol = 1e-2, atol = 1e-2`. Inputs are fp32 (per
task spec); fp32 accumulators on the Matrix Cores.

For every shape we report two numbers:

- **Reference (PyTorch eager, µs)** — the realistic upper-time baseline using
  `F.conv1d` for `causal_conv1d` and explicit `torch.matmul` chunk-loops for
  the gated DeltaNet kernels. Same fp32 inputs, same protocol.
- **Optimized (Triton AMD, µs)** — the kernel in `kernels/<name>/kernel.py`
  on this repo's tuned configs.

`Speedup = reference / optimized`. All numbers are min-of-50 microbenchmarks
(5 warmup + 50 timed, with `torch.cuda.Event` pairs and an L2-flush between
iterations).

Raw data: [`results/sweep_20260510_192629.csv`](../results/sweep_20260510_192629.csv),
[`results/baseline_compare.csv`](../results/baseline_compare.csv).

### 3.1 `causal_conv1d` — depthwise causal 1D convolution

Configs autotuned per shape. Insight: this kernel is memory-bound, so small
`(BLOCK_S=64, BLOCK_D=16)` tiles win because they expose more programs across
the 304 CUs than fewer-big-tiles does. Hand-picked configs (`BLOCK_S=256,
BLOCK_D=64`) were left ~30-39% on the table.

| Shape `(B, D, S, W)` | Reference (µs) | Optimized (µs) | Speedup |
|---|---:|---:|---:|
| (1, 1536, 2048, 4) | 74.45 | 33.20 | **2.24×** |
| (1, 2560, 2048, 4) | 89.32 | 37.17 | **2.40×** |
| (1, 2560, 4096, 4) | 130.34 | 50.23 | **2.59×** |
| **geomean** | **96.1** | **39.4** | **2.41×** |

### 3.2 `chunk_fwd_h` — gated DeltaNet inter-chunk state recurrence

Per-shape `num_warps`/`num_stages` autotuned (small wins of 0–4% — original
hand-picked configs were near-optimal). The wide range of speedups reflects
how much faster the Triton kernel scales than an eager Python-loop reference
as `T` grows.

| Shape `(B, T, H, K, V)` | Reference (µs) | Optimized (µs) | Speedup |
|---|---:|---:|---:|
| (1, 64, 1, 64, 64) | 118.75 | 31.35 | **3.79×** |
| (2, 512, 3, 64, 64) | 747.26 | 40.97 | **18.24×** |
| (2, 1024, 3, 64, 64) | 1434.59 | 50.43 | **28.44×** |
| **geomean** | **480.6** | **40.1** | **12.52×** |

### 3.3 `chunk_fwd_o` — gated DeltaNet chunkwise output

Hand-picked configs were optimal in the 9-config sweep — no autotune-driven
change.

| Shape `(B, T, H, K, V)` | Reference (µs) | Optimized (µs) | Speedup |
|---|---:|---:|---:|
| (1, 64, 1, 64, 64) | 153.35 | 47.51 | **3.23×** |
| (2, 512, 3, 64, 64) | 185.78 | 67.75 | **2.74×** |
| (2, 1024, 3, 64, 64) | 178.73 | 71.68 | **2.49×** |
| **geomean** | **172.3** | **61.7** | **2.80×** |

### 3.4 `recompute_w_u` — gated DeltaNet WY-transform recompute

Persistent-blocked launch. Per-shape `num_warps`, `num_stages`, and `GROUP_SIZE`
autotuned over a 27-config grid. Insight: `num_warps=4` (vs the hand-picked 8)
wins on every shape — `num_warps=4 × 64-lane wavefronts = 256 threads/CTA`,
exactly the size of the 64×64 MFMA tile, so the warps are perfectly utilized
without idle lanes. `GROUP_SIZE=16` also helps the small/medium shapes by
broadening the L2-friendly tile-reorder window. Improvement +17-26% per shape.

| Shape `(B, T, H, K, V)` | Reference (µs) | Optimized (µs) | Speedup |
|---|---:|---:|---:|
| (1, 64, 1, 64, 64) | 98.30 | 23.25 | **4.23×** |
| (2, 512, 3, 64, 64) | 126.57 | 42.14 | **3.00×** |
| (2, 1024, 3, 64, 64) | 137.03 | 44.22 | **3.10×** |
| **geomean** | **119.7** | **35.1** | **3.40×** |

### 3.5 Sanity check — vs `torch.compile(mode="max-autotune-no-cudagraphs")`

The Triton kernels also outperform PyTorch's own auto-compiled path (which
itself emits Triton-AMD code under the hood) on every shape:

| Kernel | Geomean speedup over `torch.compile` |
|---|---:|
| `causal_conv1d` | **2.87×** |
| `chunk_fwd_h` | **4.27×** |
| `chunk_fwd_o` | **1.57×** |
| `recompute_w_u` | **2.74×** |

### 3.6 Reproducing the autotune

```bash
# On the MI300X box (single VF, ROCm 7.x, PyTorch ROCm 6.2):
python benchmarks/autotune.py --kernels all --mode bench
# writes results/autotune_<kernel>.json + results/autotune_summary.csv
# total wall time on a 304-CU MI300X VF: ~110s for 192 configs.

python benchmarks/pytorch_baseline.py
# writes results/baseline_compare.csv with Triton vs eager vs torch.compile.
```

---

## 4. How to reproduce

All commands assume the working directory is the repository root and that
PyTorch ROCm 6.2 (or newer) wheels and Triton 3.1+ with the AMD backend are
already installed.

### 4.1 Single kernel — correctness + benchmark

```bash
# Run correctness then bench for one kernel:
python eval.py both kernels/causal_conv1d/
python eval.py both kernels/chunk_fwd_h/
python eval.py both kernels/chunk_fwd_o/
python eval.py both kernels/recompute_w_u/
```

`eval.py` accepts `correctness`, `bench`, or `both` as the first
positional argument. `both` runs correctness first and skips the bench
on `fail-correctness`. The bench writes a CSV to `results/<kernel>.csv`
matching the columns of the tables in §3.

### 4.2 Full sweep across all four kernels

```bash
# Drives all four kernels' canonical shape lists end-to-end:
python run_sweep.py
```

`run_sweep.py` calls `eval.py both` for each kernel directory under
`kernels/` and aggregates the results into `results/sweep.csv`.

### 4.3 GPU monitoring during the run

In a second terminal, alongside any of the bench commands above:

```bash
bash scripts/monitor_gpu.sh
```

This tails `rocm-smi --csv` so the operator can confirm clocks are at
boost and HBM utilization is in the expected range during the timed
loop. The script writes to stdout; redirect to a file if a record is
needed for a given run.

### 4.4 Sanity checks before timing

Before publishing numbers, the operator should confirm:

- `rocm-smi --showperflevel` reports `high` (or `manual` if clocks are
  pinned).
- `rocm-smi --showpids` reports only the bench process.
- `python eval.py correctness kernels/<name>/` passes for the kernel
  being measured.
- The reported row was produced with the `--seed 0` default (see §1.3) so
  numbers are reproducible across machines and runs.

---

## 5. References

- AMD, *AMD Instinct MI300X Platform Architecture Whitepaper*,
  https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- AMD ROCm documentation, https://rocm.docs.amd.com
- AMD, `rocm-smi` reference,
  https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/
- PyTorch ROCm install guide,
  https://pytorch.org/get-started/locally/ (select ROCm 6.2 / 7.x).
- Triton AMD backend overview,
  https://triton-lang.org/main/programming-guide/chapter-3/amdgpu.html
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with
  Delta Rule*. arXiv:2412.06464 (2024).
