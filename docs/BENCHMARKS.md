# Benchmarks

This document specifies the timing methodology, hardware setup, correctness
threshold, and reproduction commands used to measure all four kernels in this
repository. Results tables below are intentionally left as `<pending>` cells
to be filled in by `python eval.py both kernels/<name>/` runs.

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
7. **Status column**: `pass` if correctness is within tolerance and the
   kernel completed all 50 iterations without error, `fail-correctness`
   if correctness check missed tolerance, `fail-runtime` on any kernel
   launch failure, `<pending>` until measured.

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

## 3. Results

Measured on AMD Instinct MI300X VF (gfx942), single virtual function visible to PyTorch.
Shapes match `kernels/<name>/task.yml` exactly. All correctness checks pass at `rtol=1e-2, atol=1e-2`.
Inputs are float32 (matching task spec); fp32 accumulators on the Matrix Cores.

Raw CSV: `results/sweep_20260510_192629.csv`.

### 3.1 causal_conv1d

Depthwise 1D causal convolution. 5/5 test shapes pass correctness.

**Tests** (correctness, max\|diff\|):

| Shape `(B, D, S, W)` | max\|diff\| | status |
|---|---:|---|
| (1, 64, 64, 4) | 9.5e-7 | PASS |
| (2, 128, 128, 4) | 1.9e-6 | PASS |
| (1, 256, 256, 3) | 9.5e-7 | PASS |
| (1, 128, 64, 8) | 1.9e-6 | PASS |
| (4, 64, 128, 4) | 9.5e-7 | PASS |

**Benchmarks** (5 warmup + 50 timed iters):

| Shape `(B, D, S, W)` | min_us | p50_us | mean_us |
|---|---:|---:|---:|
| (1, 1536, 2048, 4) | **28.18** | 30.75 | 32.01 |
| (1, 2560, 2048, 4) | **33.80** | 34.08 | 34.73 |
| (1, 2560, 4096, 4) | **49.55** | 50.55 | 51.17 |

### 3.2 chunk_fwd_h

Gated DeltaNet inter-chunk state recurrence. 3/3 test shapes pass correctness.

**Tests:**

| Shape `(B, T, H, K, V)` | max\|diff\| | status |
|---|---:|---|
| (1, 64, 1, 64, 64) | 3.6e-7 | PASS |
| (2, 128, 4, 64, 64) | 7.2e-7 | PASS |
| (1, 256, 4, 64, 128) | 1.4e-6 | PASS |

**Benchmarks:**

| Shape `(B, T, H, K, V)` | min_us | p50_us | mean_us |
|---|---:|---:|---:|
| (1, 64, 1, 64, 64) | **35.08** | 39.29 | 41.08 |
| (2, 512, 3, 64, 64) | **28.51** | 35.48 | 39.70 |
| (2, 1024, 3, 64, 64) | **35.40** | 35.60 | 36.53 |

### 3.3 chunk_fwd_o

Gated DeltaNet chunkwise output (4 dots per block, single-pass). 3/3 test shapes pass.

**Tests:**

| Shape `(B, T, H, K, V)` | max\|diff\| | status |
|---|---:|---|
| (1, 64, 1, 64, 64) | 1.7e-5 | PASS |
| (2, 128, 4, 64, 64) | 1.5e-5 | PASS |
| (1, 256, 4, 64, 128) | 1.9e-5 | PASS |

**Benchmarks:**

| Shape `(B, T, H, K, V)` | min_us | p50_us | mean_us |
|---|---:|---:|---:|
| (1, 64, 1, 64, 64) | **36.44** | 40.57 | 42.19 |
| (2, 512, 3, 64, 64) | **39.73** | 40.81 | 43.11 |
| (2, 1024, 3, 64, 64) | **43.22** | 44.38 | 45.69 |

### 3.4 recompute_w_u

Gated DeltaNet WY-transform recompute (two GEMMs per chunk). 3/3 test shapes pass.

**Tests:**

| Shape `(B, T, H, K, V)` | status |
|---|---|
| (1, 64, 2, 64, 64) | PASS |
| (2, 128, 4, 64, 64) | PASS |
| (1, 256, 4, 64, 128) | PASS |

**Benchmarks:**

| Shape `(B, T, H, K, V)` | min_us | p50_us | mean_us |
|---|---:|---:|---:|
| (1, 64, 1, 64, 64) | **34.88** | 37.45 | 38.38 |
| (2, 512, 3, 64, 64) | **40.81** | 43.06 | 44.47 |
| (2, 1024, 3, 64, 64) | **36.16** | 41.37 | 42.40 |

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
- The `<pending>` row being filled has run with the `--seed 0` default
  (see §1.3).

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
