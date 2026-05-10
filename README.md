<div align="center">

# Theorem

**AMD Instinct MI300X-optimized GPU kernels for transformer-relevant ops.**

CDNA3-tuned Triton kernels for causal 1-D convolution and the gated DeltaNet
chunkwise primitives. Hand-tuned, autotune-swept, and reproducible end-to-end
on a single MI300X.

[Demo](#demo)&nbsp;·&nbsp;[Slides](assets/theorem_slides.pdf)&nbsp;·&nbsp;[Benchmarks](docs/BENCHMARKS.md)&nbsp;·&nbsp;[Optimizations](docs/OPTIMIZATIONS.md)&nbsp;·&nbsp;[Architecture](docs/ARCHITECTURE.md)

</div>

---

## Headline

Geomean speedups vs PyTorch eager fp32 across each kernel's three benchmark shapes on a single AMD Instinct MI300X (gfx942). Full per-shape numbers and methodology in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

<div align="center">

| `causal_conv1d` | `chunk_fwd_h` | `chunk_fwd_o` | `recompute_w_u` |
|:---:|:---:|:---:|:---:|
| **2.73×** | **12.42×** | **4.51×** | **2.96×** |

</div>

The same kernels also outperform `torch.compile(mode="max-autotune-no-cudagraphs")` — which itself emits Triton-AMD code — by **2.34× to 4.10× geomean**.

---

## Demo

<div align="center">

<video src="https://github.com/yhinai/Theorem/raw/main/assets/demo.mp4" controls width="720">
  Your browser does not display the video. Download it:
  <a href="assets/demo.mp4">assets/demo.mp4</a>.
</video>

[▶ Watch / Download `assets/demo.mp4`](assets/demo.mp4)

</div>

---

## Slides

[`assets/theorem_slides.pdf`](assets/theorem_slides.pdf) — the talk-track companion with the architecture diagram and per-kernel optimization breakdowns.

---

## What this is

Theorem is a deliberately narrow collection of GPU kernels: take a small set of ops on the hot path of modern sub-quadratic sequence models (Mamba/Mamba-2, gated DeltaNet) and squeeze them on the MI300X. No generality, no portability shims — just kernels that know exactly what hardware they are running on.

The optimization story is **measured, not asserted.** Every config in every kernel was chosen by an autotune sweep on the actual hardware, every speedup number on this page is reproducible by `python benchmarks/autotune.py && python benchmarks/pytorch_baseline.py`, and the raw CSVs are committed in [`results/`](results/).

---

## Hardware & software target

<div align="center">

| Hardware | | Software |
|---|---|---|
| Device | AMD Instinct MI300X | PyTorch + ROCm 6.2 wheels |
| Architecture | CDNA3 (`gfx942`) | Triton ≥ 3.1 (AMD backend) |
| Compute units | 304 | NumPy, PyYAML |
| Wavefront | 64 lanes | ROCm 7.x runtime |
| Memory | ~192 GB HBM3e | Python ≥ 3.11 |

</div>

Earlier ROCm versions are not supported and will not be backported.

---

## Kernel inventory

<table>
<thead>
<tr>
  <th>Kernel</th>
  <th>What it does</th>
  <th>Reference (µs)</th>
  <th>Optimized (µs)</th>
  <th>Speedup</th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>causal_conv1d</code></td>
  <td>Depthwise 1D causal convolution. Used in Mamba / Mamba-2-style architectures. Memory-bound; small <code>(64×16)</code> tiles win because they expose more programs across the 304 CUs than fewer big tiles do.</td>
  <td align="right">95.0</td>
  <td align="right">34.6</td>
  <td align="right"><strong>2.73×</strong></td>
</tr>
<tr>
  <td><code>chunk_fwd_h</code></td>
  <td>Gated DeltaNet inter-chunk recurrence <code>S<sub>c+1</sub> = G<sub>c</sub>·S<sub>c</sub> + K<sub>c</sub><sup>T</sup>V<sub>c</sub></code>. State pinned in registers across the chunk loop; <code>tl.dot</code> mapped to Matrix Cores.</td>
  <td align="right">489.9</td>
  <td align="right">39.4</td>
  <td align="right"><strong>12.42×</strong></td>
</tr>
<tr>
  <td><code>chunk_fwd_o</code></td>
  <td>Gated DeltaNet chunkwise output (local causal attention + global state). The biggest single tuning win: <code>num_warps=16→4</code> + <code>matrix_instr_nonkdim=16</code> picks the 16×16×4 fp32 MFMA shape that matches the 64×64 chunk geometry.</td>
  <td align="right">192.7</td>
  <td align="right">42.7</td>
  <td align="right"><strong>4.51×</strong></td>
</tr>
<tr>
  <td><code>recompute_w_u</code></td>
  <td>Gated DeltaNet WY-transform recompute (two GEMMs per chunk). Persistent-blocked launch, L2 reordering, autotuned <code>num_warps=4</code>: 4 × 64-lane wavefronts = 256 threads/CTA — exactly right for the 64×64 MFMA tile.</td>
  <td align="right">124.4</td>
  <td align="right">42.1</td>
  <td align="right"><strong>2.96×</strong></td>
</tr>
</tbody>
</table>

Full per-shape tables with min / p50 / mean and the comparison against `torch.compile`: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Quick start

```bash
git clone https://github.com/yhinai/Theorem.git
cd Theorem
bash scripts/setup_env.sh        # creates .venv, installs torch (ROCm 6.2 wheels), triton, deps
source .venv/bin/activate
python scripts/run_amd.py        # smoke-test all four kernels on the smallest test shape
```

Expected output: four `PASS` lines and a one-line GPU banner.

---

## Optimization principles

Four patterns repeat across every kernel — written up once in [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md), summarized here.

- **Wavefront-aware block sizing.** Block sizes are multiples of 64 along the contiguous axis. The classic NVIDIA "more warps = faster" intuition is wrong on CDNA3: `num_warps=16` over-subscribes (1024 threads/CTA) when the 64×64 MFMA tile only needs 256.
- **LDS pipelining via `num_stages`.** Inner-reduction loops set `num_stages ≥ 2` so the next tile's HBM3e load overlaps the current tile's MFMA. Per-shape autotuned — too high pressures LDS, too low serializes memory.
- **MFMA tile shape (`matrix_instr_nonkdim`).** The AMD backend's MFMA selector. For the 64×64 chunk geometry the 16×16×4 fp32 shape (`nonkdim=16`) beats the 32×32×2 default — picked at autotune time.
- **Per-shape configuration tuning.** Configs live in `SHAPE_CONFIGS` dicts at module load time. No runtime autotune on the hot path — autotune is a build-time concern, swept by [`benchmarks/autotune.py`](benchmarks/autotune.py).

---

## Optimization journey — what shipped, what didn't

Three rounds of work, three insights worth keeping:

| Round | Approach | Outcome |
|---|---|---|
| 1 | Sweep `BLOCK_*` × `num_warps` × `num_stages` for all shape-aware kernels | ✅ `causal_conv1d` +30-39% per shape (small tiles beat big ones) |
| 2 | Refactor `recompute_w_u` to a dict-keyed `SHAPE_CONFIGS` then sweep | ✅ +17-26% per shape (`num_warps=4` beats hand-picked 8) |
| 3 | Add `matrix_instr_nonkdim` to the matmul kernel sweeps | ✅ `chunk_fwd_o` +47% on the larger shapes |
| ✗ | **Fuse `chunk_fwd_h + chunk_fwd_o`** to keep `h` in registers across the 4 dots | Faster on smallest shape (1.65×), slower on larger ones — the unfused pair has 16-32× more parallelism than the per-(B,H) fused loop |
| ✗ | **LDS-stage `causal_conv1d`** input tile across the W taps | Triton 3.1 on AMD couldn't slice a wide tile per-`j` without a `tl.where` workaround that ate the savings |

Both negative results are documented honestly because the *constraint* matters more than the *configuration*: AMD CDNA3 isn't NVIDIA, and what works at the algorithmic level on Hopper-style hardware doesn't always transfer to a 304-CU chip with 64-lane wavefronts.

---

## Reproducing the numbers

```bash
# Smoke test (smallest shape per kernel, ~5s):
python scripts/run_amd.py

# Per-kernel correctness + benchmark:
python eval.py both kernels/causal_conv1d/

# Cross-kernel sweep -> results/sweep_<timestamp>.csv:
python run_sweep.py --mode both

# Triton config autotune (writes results/autotune_*.json + summary.csv):
python benchmarks/autotune.py --kernels all --mode bench

# Triton vs PyTorch eager vs torch.compile (writes results/baseline_compare.csv):
python benchmarks/pytorch_baseline.py

# GPU telemetry during a run:
bash scripts/monitor_gpu.sh /tmp/gpu_telemetry.csv &
```

Methodology, timing protocol, and tolerance constants live in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Raw outputs are committed under [`results/`](results/) so the headline numbers can be checked against the source data.

---

## Repo layout

```text
Theorem/
├── kernels/                       4 kernel modules (kernel.py + reference.py + task.yml + README.md)
│   ├── causal_conv1d/
│   ├── chunk_fwd_h/
│   ├── chunk_fwd_o/
│   └── recompute_w_u/
├── benchmarks/
│   ├── autotune.py                per-shape Triton config sweep
│   ├── pytorch_baseline.py        Triton vs eager vs torch.compile
│   └── apply_best_configs.py      writes best configs back into kernel.py
├── eval.py                        single-kernel correctness + benchmark
├── run_sweep.py                   cross-kernel correctness + benchmark sweep
├── utils.py                       allclose / device probes / lazy import
├── scripts/
│   ├── run_amd.py                 smoke-test all 4 kernels
│   ├── monitor_gpu.sh             rocm-smi telemetry to CSV
│   ├── cpu_reference.py           NumPy oracle for causal_conv1d
│   └── setup_env.sh               one-shot ROCm venv + torch install
├── docs/
│   ├── ARCHITECTURE.md            CDNA3 mental model + repo shape
│   ├── OPTIMIZATIONS.md           per-kernel optimization deep-dive
│   └── BENCHMARKS.md              methodology + per-shape result tables
├── results/                       raw CSVs from runs (committed)
├── assets/
│   ├── demo.mp4                   the demo video at the top of this README
│   └── theorem_slides.pdf         the slide deck
└── .github/workflows/ci.yml       syntax + task.yml validation (no GPU runner yet)
```

---

## Citations

- AMD MI300X architecture brief — <https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html>
- ROCm documentation — <https://rocm.docs.amd.com/>
- Triton AMD backend — <https://triton-lang.org/main/dialects/amdgpu.html>
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule* (arXiv:2412.06464) — <https://arxiv.org/abs/2412.06464>

---

## License

MIT — see [`LICENSE`](LICENSE).
