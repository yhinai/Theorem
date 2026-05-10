"""PyTorch baseline benchmark harness for Triton kernels on AMD MI300X.

Compares the four custom Triton kernels in this repo against two PyTorch
baselines per kernel:

    eager_baseline    -- plain PyTorch (the existing reference implementation,
                         or a slightly tightened version of it). On ROCm this
                         lands on rocBLAS / MIOpen / eager ROCm dispatchers and
                         is the "lower bound" of framework-only performance.

    compiled_baseline -- the same eager function wrapped with
                         `torch.compile(..., mode="max-autotune-no-cudagraphs")`.
                         On a ROCm install this typically lowers through the
                         Triton AMD backend and is the "upper bound" of pure
                         framework code with no hand-written kernel.

For each benchmark shape declared in each kernel's `task.yml`, we time:

    * the existing Triton `custom_kernel`
    * eager_baseline
    * compiled_baseline

with 5 warmup + 50 timed iterations using `torch.cuda.Event` pairs around a
`torch.cuda.synchronize()` (PyTorch's public CUDA API is the supported
on-ramp to the AMD device under ROCm; the device is still addressed as
`cuda:0` on the public API).

Output:

    * a markdown table on stdout
    * `<repo>/results/baseline_compare.csv`

CLI:

    python benchmarks/pytorch_baseline.py
    python benchmarks/pytorch_baseline.py --kernels causal_conv1d,chunk_fwd_h

Per-cell failures (a baseline OOMs, torch.compile bails, etc.) are caught and
recorded as NaN so a single broken cell does not break the whole sweep.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project root on sys.path so `from kernels.<name> import ...` works
# regardless of where the script is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402


WARMUP_ITERS = 5
TIMED_ITERS = 50
DEVICE = "cuda:0"  # AMD device under ROCm exposes itself via the cuda API


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_callable(fn: Callable[[], Any]) -> float:
    """Return min latency in microseconds across TIMED_ITERS runs.

    Uses GPU event pairs around the callable; synchronizes once before warmup
    starts and once after timing ends. We take the minimum (not the mean) so
    one-off scheduler hiccups do not dominate the headline number — this
    matches what `run_sweep.py` already does for the Triton-vs-reference
    sweep.
    """
    # Warmup
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()

    times_ms: List[float] = []
    for _ in range(TIMED_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return min(times_ms) * 1_000.0  # ms -> us


def _safe_time(label: str, shape: Dict[str, Any], fn: Callable[[], Any]) -> float:
    """Wrap _time_callable so per-cell failures don't abort the sweep."""
    try:
        return _time_callable(fn)
    except Exception as exc:  # noqa: BLE001 — we genuinely want everything
        print(
            f"  [warn] {label} failed on shape {shape}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return float("nan")


# ---------------------------------------------------------------------------
# Per-kernel eager baselines
#
# Each returns a *callable taking no args* that runs the baseline once on
# pre-staged tensors. We build the inputs once per shape, then time the
# callable repeatedly to amortize input construction out of the measurement.
# ---------------------------------------------------------------------------

def _causal_conv1d_eager(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Plain depthwise causal conv1d: left zero-pad + grouped conv."""
    B, D, S = x.shape
    W = weight.shape[1]
    x_padded = F.pad(x, (W - 1, 0))
    w = weight.unsqueeze(1)
    return F.conv1d(x_padded, w, bias=bias, groups=D)


def _chunk_fwd_h_eager(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                       B: int, T: int, H: int, K: int, V: int, BT: int) -> torch.Tensor:
    """Eager per-chunk recurrence, vectorized over (batch, head).

    Matches the math of `kernels/chunk_fwd_h/reference.py` but with the
    per-(b, h) Python loop collapsed into batched ops — this is what a
    competent PyTorch user would write before reaching for Triton.
    """
    NT = T // BT
    device = k.device
    dtype = torch.float32

    # [B, NT, BT, H, K] / [B, NT, BT, H, V] / [B, NT, BT, H]
    k_c = k.reshape(B, NT, BT, H, K)
    v_c = v.reshape(B, NT, BT, H, V)
    g_c = g.reshape(B, NT, BT, H)

    h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
    state = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    for c in range(NT):
        k_chunk = k_c[:, c]           # [B, BT, H, K]
        v_chunk = v_c[:, c]           # [B, BT, H, V]
        g_chunk = g_c[:, c]           # [B, BT, H]

        g_end = g_chunk[:, -1, :]     # [B, H]
        diff = g_end.unsqueeze(1) - g_chunk             # [B, BT, H]
        v_gated = v_chunk * torch.exp(diff).unsqueeze(-1)   # [B, BT, H, V]

        # k_chunk: [B, BT, H, K] -> [B, H, K, BT]; v_gated: [B, BT, H, V] -> [B, H, BT, V]
        k_t = k_chunk.permute(0, 2, 3, 1)
        v_t = v_gated.permute(0, 2, 1, 3)
        update = torch.matmul(k_t, v_t)                      # [B, H, K, V]

        decay = torch.exp(g_end).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        state = state * decay + update
        h_out[:, c] = state

    return h_out


def _chunk_fwd_o_eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       g: torch.Tensor, h: torch.Tensor, scale: float,
                       B: int, T: int, H: int, K: int, V: int, BT: int) -> torch.Tensor:
    """Eager chunkwise output: per-chunk causal qk + global state hop.

    Vectorized over (B, NT, H) to keep this realistic. Inside each chunk we
    still build the [BT, BT] qk tile.
    """
    NT = T // BT
    device = q.device

    q_c = q.reshape(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)   # [B, NT, H, BT, K]
    k_c = k.reshape(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)   # [B, NT, H, BT, K]
    v_c = v.reshape(B, NT, BT, H, V).permute(0, 1, 3, 2, 4)   # [B, NT, H, BT, V]
    g_c = g.reshape(B, NT, BT, H).permute(0, 1, 3, 2)          # [B, NT, H, BT]
    # h: [B, NT, H, K, V]

    eg = torch.exp(g_c).unsqueeze(-1)            # [B, NT, H, BT, 1]
    emg = torch.exp(-g_c).unsqueeze(-1)          # [B, NT, H, BT, 1]
    q_gated = q_c * eg
    v_gated = v_c * emg

    qk = torch.matmul(q_gated, k_c.transpose(-1, -2))    # [B, NT, H, BT, BT]

    idx = torch.arange(BT, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)        # [BT, BT]
    qk = qk * causal

    local_out = torch.matmul(qk, v_gated)                # [B, NT, H, BT, V]
    global_out = torch.matmul(q_gated, h)                # [B, NT, H, BT, V]
    out_chunks = scale * (local_out + global_out)        # [B, NT, H, BT, V]

    # [B, NT, H, BT, V] -> [B, T, H, V]
    return out_chunks.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).contiguous()


def _recompute_w_u_eager(data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager (w, u) recompute: two batched matmuls per chunk."""
    BT = 64
    k, v, beta, A, g = data.k, data.v, data.beta, data.A, data.g
    B, T, H, K = k.shape
    V = v.shape[-1]
    n_chunks = T // BT

    k_c = k.reshape(B, n_chunks, BT, H, K).permute(0, 1, 3, 2, 4)
    v_c = v.reshape(B, n_chunks, BT, H, V).permute(0, 1, 3, 2, 4)
    beta_c = beta.reshape(B, n_chunks, BT, H).permute(0, 1, 3, 2)
    g_c = g.reshape(B, n_chunks, BT, H).permute(0, 1, 3, 2)
    A_c = A.reshape(B, n_chunks, BT, H, BT).permute(0, 1, 3, 2, 4)

    v_scaled = v_c * beta_c.unsqueeze(-1)
    k_scaled = k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1)

    u_c = torch.matmul(A_c, v_scaled)
    w_c = torch.matmul(A_c, k_scaled)

    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).contiguous()
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).contiguous()
    return w, u


# ---------------------------------------------------------------------------
# Kernel registry: how to build inputs and how to run each baseline
# ---------------------------------------------------------------------------

def _stage_causal_conv1d(shape: Dict[str, Any]):
    from kernels.causal_conv1d import custom_kernel
    from kernels.causal_conv1d.reference import generate_input

    data = generate_input(shape["B"], shape["D"], shape["S"], shape["W"], shape["seed"])
    x, weight, bias = data

    def run_triton():
        return custom_kernel(data)

    def run_eager():
        return _causal_conv1d_eager(x, weight, bias)

    compiled = torch.compile(_causal_conv1d_eager, mode="max-autotune-no-cudagraphs")

    def run_compiled():
        return compiled(x, weight, bias)

    return run_triton, run_eager, run_compiled


def _stage_chunk_fwd_h(shape: Dict[str, Any]):
    from kernels.chunk_fwd_h import custom_kernel
    from kernels.chunk_fwd_h.reference import generate_input, CHUNK_SIZE

    data = generate_input(shape["B"], shape["T"], shape["H"], shape["K"], shape["V"], shape["seed"])
    k, v, g = data["k"], data["v"], data["g"]
    B, T, H, K, V = data["B"], data["T"], data["H"], data["K"], data["V"]
    BT = CHUNK_SIZE

    def run_triton():
        return custom_kernel(data)

    def run_eager():
        return _chunk_fwd_h_eager(k, v, g, B, T, H, K, V, BT)

    compiled = torch.compile(_chunk_fwd_h_eager, mode="max-autotune-no-cudagraphs")

    def run_compiled():
        return compiled(k, v, g, B, T, H, K, V, BT)

    return run_triton, run_eager, run_compiled


def _stage_chunk_fwd_o(shape: Dict[str, Any]):
    from kernels.chunk_fwd_o import custom_kernel
    from kernels.chunk_fwd_o.reference import generate_input

    data = generate_input(shape["B"], shape["T"], shape["H"], shape["K"], shape["V"], shape["seed"])
    q, k, v, g, h = data["q"], data["k"], data["v"], data["g"], data["h"]
    scale = float(data["scale"])
    B, T, H, K, V, BT = data["B"], data["T"], data["H"], data["K"], data["V"], data["BT"]

    def run_triton():
        return custom_kernel(data)

    def run_eager():
        return _chunk_fwd_o_eager(q, k, v, g, h, scale, B, T, H, K, V, BT)

    compiled = torch.compile(_chunk_fwd_o_eager, mode="max-autotune-no-cudagraphs")

    def run_compiled():
        return compiled(q, k, v, g, h, scale, B, T, H, K, V, BT)

    return run_triton, run_eager, run_compiled


def _stage_recompute_w_u(shape: Dict[str, Any]):
    from kernels.recompute_w_u import custom_kernel
    from kernels.recompute_w_u.reference import generate_data

    data = generate_data(shape["B"], shape["T"], shape["H"], shape["K"], shape["V"], shape["seed"])

    def run_triton():
        return custom_kernel(data)

    def run_eager():
        return _recompute_w_u_eager(data)

    compiled = torch.compile(_recompute_w_u_eager, mode="max-autotune-no-cudagraphs")

    def run_compiled():
        return compiled(data)

    return run_triton, run_eager, run_compiled


KERNELS: Dict[str, Callable[[Dict[str, Any]], Tuple[Callable, Callable, Callable]]] = {
    "causal_conv1d": _stage_causal_conv1d,
    "chunk_fwd_h": _stage_chunk_fwd_h,
    "chunk_fwd_o": _stage_chunk_fwd_o,
    "recompute_w_u": _stage_recompute_w_u,
}


# ---------------------------------------------------------------------------
# Shape loading
# ---------------------------------------------------------------------------

def _load_shapes(kernel_name: str) -> List[Dict[str, Any]]:
    task_path = _REPO_ROOT / "kernels" / kernel_name / "task.yml"
    with task_path.open() as fh:
        spec = yaml.safe_load(fh)
    return list(spec.get("shapes", {}).get("benchmarks", []) or [])


def _shape_str(shape: Dict[str, Any]) -> str:
    skip = {"seed"}
    return ",".join(f"{k}={v}" for k, v in shape.items() if k not in skip)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return "nan" if (x != x) else f"{x:.2f}"  # x != x catches NaN


def _ratio(a: float, b: float) -> float:
    if a != a or b != b or b == 0:
        return float("nan")
    return a / b


def _gap(triton: float, compiled: float) -> float:
    if triton != triton or compiled != compiled or compiled == 0:
        return float("nan")
    return (triton - compiled) / compiled * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels",
        type=str,
        default=",".join(KERNELS.keys()),
        help="comma-separated list of kernels to benchmark",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(_REPO_ROOT / "results" / "baseline_compare.csv"),
        help="path to write CSV results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: no GPU device visible (need ROCm-built PyTorch on an AMD host)",
              file=sys.stderr)
        return 2

    requested = [s.strip() for s in args.kernels.split(",") if s.strip()]
    unknown = [n for n in requested if n not in KERNELS]
    if unknown:
        print(f"error: unknown kernel(s): {unknown}", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []

    for kname in requested:
        print(f"\n=== {kname} ===")
        try:
            shapes = _load_shapes(kname)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] could not load shapes: {exc}", file=sys.stderr)
            continue

        stage_fn = KERNELS[kname]
        for shape in shapes:
            print(f"  shape: {_shape_str(shape)}")
            try:
                run_triton, run_eager, run_compiled = stage_fn(shape)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] staging failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue

            t_us = _safe_time("triton", shape, run_triton)
            e_us = _safe_time("eager", shape, run_eager)
            c_us = _safe_time("compiled", shape, run_compiled)

            rows.append({
                "kernel": kname,
                "shape": _shape_str(shape),
                "triton_min_us": t_us,
                "eager_min_us": e_us,
                "compiled_min_us": c_us,
                "triton_speedup_vs_eager": _ratio(e_us, t_us),
                "gap_to_compiled_pct": _gap(t_us, c_us),
            })

            # Free callables / tensors before next shape.
            del run_triton, run_eager, run_compiled
            torch.cuda.empty_cache()

    # CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "kernel", "shape",
            "triton_min_us", "eager_min_us", "compiled_min_us",
            "triton_speedup_vs_eager", "gap_to_compiled_pct",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {csv_path}")

    # Markdown
    print("\n| kernel | shape | triton_min_us | eager_min_us | compiled_min_us | "
          "triton_speedup_vs_eager | gap_to_compiled |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        gap = row["gap_to_compiled_pct"]
        gap_str = "nan" if gap != gap else f"{gap:+.1f}%"
        speedup = row["triton_speedup_vs_eager"]
        speedup_str = "nan" if speedup != speedup else f"{speedup:.2f}x"
        print(
            f"| {row['kernel']} | {row['shape']} | "
            f"{_fmt(row['triton_min_us'])} | "
            f"{_fmt(row['eager_min_us'])} | "
            f"{_fmt(row['compiled_min_us'])} | "
            f"{speedup_str} | {gap_str} |"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
