# Theorem — AMD MI300X-optimized GPU kernels

> CDNA3-tuned Triton kernels for transformer-relevant ops: causal 1D convolution and gated DeltaNet chunkwise primitives, written for the AMD Instinct MI300X.

Theorem is a focused collection of GPU kernels hand-tuned for AMD's CDNA3
architecture. The goal is unapologetically narrow: take a small set of ops that
sit on the hot path of modern sub-quadratic sequence models (Mamba/Mamba-2,
gated DeltaNet) and squeeze them on the MI300X. No generality, no portability
shims — just kernels that know exactly what hardware they are running on.

---

## Hardware target

| Property | Value |
| --- | --- |
| Device | AMD Instinct MI300X |
| Architecture | CDNA3 (`gfx942`) |
| Compute Units | 304 |
| Wavefront width | 64 lanes |
| Memory | ~192 GB HBM3e |
| Stack | ROCm 7.x, HIP, Triton (AMD backend) |

All kernels assume `gfx942` and ROCm 7.x. Earlier ROCm versions are not
supported and will not be backported.

---

## Tooling

- **PyTorch + ROCm** — host-side tensor management and reference implementations.
- **Triton (AMD backend)** — kernel authoring; emits HIP-compatible binaries
  through the AMD backend in upstream Triton ≥ 3.1.
- **NumPy** — small numerical reference utilities and offline analysis.
- **PyYAML** — per-shape autotune configuration files.

---

## Kernel inventory — reference vs optimized

Measured on AMD Instinct MI300X. **Reference** is the PyTorch eager implementation
(`F.conv1d`, eager DeltaNet matmul/einsum loops). **Optimized** is the Triton
kernel in this repo. Both are min-of-50-iter microbenchmarks at fp32. Geomean
speedup is across the kernel's three benchmark shapes (full per-shape table in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

| Kernel | Math (sketch) | CDNA3-aware optimization | Reference (µs, geomean) | Optimized (µs, geomean) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `causal_conv1d` | Depthwise 1D causal convolution `y[t,c] = Σ_k w[k,c] · x[t-k,c]` (used in Mamba / Mamba-2-style architectures). | Block sized to whole 64-lane wavefronts; **autotuned `BLOCK_S × BLOCK_D` per shape** — small (64×16) tiles win because they expose more programs across the 304 CUs than fewer big tiles do for this memory-bound op. | 96.1 | 39.4 | **2.41×** |
| `gated_deltanet_chunk_fwd_h` | Inter-chunk recurrence `S_{c+1} = G_c · S_c + K_cᵀ V_c` over fixed-size chunks (gated DeltaNet, arXiv:2412.06464). | State `S` pinned in registers across the chunk-step loop; `tl.dot` mapped to Matrix Cores; per-shape `num_warps`/`num_stages` autotuned. | 480.6 | 40.1 | **12.52×** |
| `gated_deltanet_chunk_fwd_o` | Chunkwise output `O_c = (Q_c K_cᵀ ⊙ M) V_c + Q_c S_c` (local causal attention plus global state read). | Two `tl.dot` blocks share one Q tile in registers; causal mask materialized at compile time per `BT`; state read coalesced from HBM3e through LDS. | 172.3 | 61.7 | **2.80×** |
| `gated_deltanet_recompute_w_u` | Recomputes the WY-transform helpers `W = β · (I − tril(K Kᵀ)·β)⁻¹` and `U` used by the backward pass. | Two `tl.dot` matmuls per chunk; persistent-blocked program scheduling; L2 reordering; **autotuned `num_warps` / `num_stages` / `GROUP_SIZE` per shape** — `num_warps=4` (vs the hand-picked 8) wins on every shape. | 119.7 | 35.1 | **3.40×** |

The Triton kernels also outperform `torch.compile(mode="max-autotune-no-cudagraphs")`
on every shape — by **1.57× to 4.27×** geomean (full data:
[`results/baseline_compare.csv`](results/baseline_compare.csv),
[`results/autotune_summary.csv`](results/autotune_summary.csv)).

> Configs were swept via `python benchmarks/autotune.py --kernels all --mode bench`
> on the MI300X. Biggest wins:
> - `causal_conv1d`: +30-39% per shape (small tiles beat big tiles for memory-bound ops on 304 CUs)
> - `recompute_w_u`: +17-26% per shape (`num_warps=4 × 64-lane wavefronts = 256` threads/CTA, exactly right for the 64×64 MFMA tile)

---

## Quick start

```bash
git clone https://github.com/yhinai/Theorem.git && cd Theorem
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_env.sh
python eval.py test kernels/causal_conv1d/
```

The PyTorch wheel must be the **ROCm** build — see `requirements.txt` for the
correct `--index-url`. `scripts/setup_env.sh` sets the ROCm and Triton
environment variables that the evaluator and sweeper expect.

---

## Repo layout

```text
Theorem/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── eval.py
├── utils.py
├── run_sweep.py
├── kernels/
│   ├── causal_conv1d/
│   ├── chunk_fwd_h/
│   ├── chunk_fwd_o/
│   └── recompute_w_u/
├── scripts/
│   ├── run_amd.py
│   ├── monitor_gpu.sh
│   ├── cpu_reference.py
│   └── setup_env.sh
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OPTIMIZATIONS.md
│   └── BENCHMARKS.md
└── results/
    └── .gitkeep
```

- `eval.py` — single entry point for correctness + microbenchmarks per kernel.
- `utils.py` — shared helpers (timing, tolerance, device probes).
- `run_sweep.py` — drives autotune sweeps across the per-shape config files.
- `kernels/<name>/` — one directory per kernel: Triton source, reference op,
  configuration YAML, and unit tests.
- `scripts/run_amd.py` — orchestrates a full benchmark run on an MI300X host.
- `scripts/monitor_gpu.sh` — wraps `rocm-smi` for power, clock, and HBM
  occupancy traces during sweeps.
- `scripts/cpu_reference.py` — NumPy reference oracles used to validate kernel
  outputs against bitwise-stable expectations.
- `scripts/setup_env.sh` — exports ROCm/Triton environment variables.
- `docs/ARCHITECTURE.md` — CDNA3 mental model and how it shapes the kernels.
- `docs/OPTIMIZATIONS.md` — the recurring optimization patterns, written up.
- `docs/BENCHMARKS.md` — methodology, shapes covered, how to read the CSVs.
- `results/` — benchmark CSV outputs (gitignored except `.gitkeep`).

---

## Optimization principles applied

These four patterns recur across every kernel in this repo. They are written
down once so each kernel doesn't have to re-derive them.

- **Wavefront-aware block sizing.** CDNA3 issues in 64-lane wavefronts. Triton
  block sizes are picked so that the contiguous axis of every load and every
  `tl.dot` is a multiple of 64 — never 32. Mismatched block shapes leave half
  the wavefront idle and burn LDS bandwidth for no work.
- **LDS pipelining via `num_stages`.** Every kernel that has an inner reduction
  loop (chunk steps in `chunk_fwd_h`, K/V steps in `chunk_fwd_o`) sets
  `num_stages ≥ 2` so the next tile's HBM3e load overlaps the current tile's
  Matrix Core dot. The exact `num_stages` is per-shape — too high and we
  pressure the LDS, too low and we serialize on memory.
- **Fused dot-accumulate on Matrix Cores.** Reductions go through `tl.dot` with
  `acc=` chaining so the AMD backend lowers them onto Matrix Core MFMA
  instructions in fp16/bf16 with fp32 accumulate. Hand-rolled `tl.sum` over the
  same axis is left as a sanity-check baseline only.
- **Per-shape configuration tuning.** `BT`, `BK`, `BV`, `num_warps`,
  `num_stages`, and waves-per-EU live in a YAML next to each kernel. The sweep
  driver picks the best config per `(batch, heads, seqlen, head_dim)` tuple and
  the kernel imports the chosen config at module-load time. No runtime
  autotune on the hot path — autotune is a build-time concern.

The full write-up of how each pattern manifests per kernel is in
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md).

---

## Reproducing benchmarks

```bash
bash scripts/monitor_gpu.sh results/gpu_trace.log &
python scripts/run_amd.py --kernels all --out results/
python run_sweep.py --kernel chunk_fwd_o --config kernels/chunk_fwd_o/configs.yaml
```

`scripts/run_amd.py` produces one CSV per kernel under `results/` with columns
`shape, mean_ms, p50_ms, p99_ms, gbps, tflops`. The methodology — warmup count,
clock locking, cache-flush pattern between iterations — is documented in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Citations & further reading

- ROCm documentation — <https://rocm.docs.amd.com/>
- Triton AMD backend — <https://triton-lang.org/main/dialects/amdgpu.html>
- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*,
  arXiv:2412.06464 — <https://arxiv.org/abs/2412.06464>
- AMD Instinct MI300X architecture brief —
  <https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html>

---

## License

MIT — see [`LICENSE`](LICENSE).
