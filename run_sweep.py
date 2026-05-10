"""Multi-kernel sweep runner for the AMD MI300X harness.

Iterates over the union of selected kernels, runs test/benchmark per kernel,
and emits a single CSV at results/sweep_<timestamp>.csv. Per-row failures are
captured to the `error` column and never abort the sweep.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from utils import (
    DeterministicContext,
    import_kernel,
    load_task,
    set_seed,
    verbose_allclose,
)

DEFAULT_KERNELS = ["causal_conv1d", "chunk_fwd_h", "chunk_fwd_o", "recompute_w_u"]
WARMUP_ITERS = 5
TIMED_ITERS = 50
DEFAULT_SEED = 0

CSV_COLUMNS = [
    "kernel", "shape", "mode",
    "B", "D_or_H", "S_or_T", "K", "V", "W",
    "min_us", "p50_us", "mean_us",
    "correct", "max_abs_diff", "error",
]


def _shape_label(shape: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items() if k != "name")


def _shape_value(shape: dict, *keys: str) -> Any:
    for k in keys:
        if k in shape:
            return shape[k]
    return ""


def _load_inputs(kernel_dir: Path, shape_kwargs: dict) -> Any:
    ref_path = Path(kernel_dir).resolve() / "reference.py"
    spec = importlib.util.spec_from_file_location(f"_ref_{Path(kernel_dir).name}", ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {ref_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen = getattr(mod, "generate_input", None)
    if gen is None:
        raise AttributeError(f"{ref_path} must export generate_input(**kwargs)")
    return gen(**shape_kwargs)


def _time_kernel(custom_kernel, args) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("AMD device unavailable — cannot benchmark")
    for _ in range(WARMUP_ITERS):
        custom_kernel(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_ITERS)]
    for i in range(TIMED_ITERS):
        starts[i].record()
        custom_kernel(*args)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(TIMED_ITERS))
    return sum(times) / len(times), times[len(times) // 2], times[0]


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return float("nan")
        if a.shape != b.shape:
            return float("nan")
        af = a.detach().to(torch.float32).cpu()
        bf = b.detach().to(torch.float32).cpu()
        finite = torch.isfinite(af) & torch.isfinite(bf)
        if not finite.any():
            return float("nan")
        return float((af - bf).abs()[finite].max())
    except Exception:
        return float("nan")


def _row(kernel: str, shape: dict, mode: str) -> dict:
    return {
        "kernel": kernel,
        "shape": _shape_label(shape),
        "mode": mode,
        "B": _shape_value(shape, "B", "batch"),
        "D_or_H": _shape_value(shape, "D", "H", "n_heads", "dim"),
        "S_or_T": _shape_value(shape, "S", "T", "seqlen", "L"),
        "K": _shape_value(shape, "K", "chunk", "chunk_size"),
        "V": _shape_value(shape, "V", "vdim"),
        "W": _shape_value(shape, "W", "width", "kernel_size"),
        "min_us": "",
        "p50_us": "",
        "mean_us": "",
        "correct": "",
        "max_abs_diff": "",
        "error": "",
    }


def _run_kernel(kernel: str, mode: str, repo_root: Path) -> list[dict]:
    rows: list[dict] = []
    kernel_dir = repo_root / "kernels" / kernel
    if not kernel_dir.exists():
        rows.append({**_row(kernel, {}, mode), "error": f"missing kernel dir: {kernel_dir}"})
        return rows
    try:
        task = load_task(kernel_dir)
        tols = task.get("tolerances", {}) or {}
        rtol = float(tols.get("rtol", 1e-3))
        atol = float(tols.get("atol", 1e-3))
        custom_kernel, ref_kernel = import_kernel(kernel_dir)
    except Exception as exc:  # noqa: BLE001
        rows.append({**_row(kernel, {}, mode), "error": f"setup-failed: {type(exc).__name__}: {exc}"})
        return rows

    if mode in ("test", "both"):
        for shape in task.get("shapes", {}).get("tests", []) or []:
            row = _row(kernel, shape, "test")
            try:
                set_seed(int(shape.get("seed", DEFAULT_SEED)))
                with DeterministicContext():
                    data = _load_inputs(kernel_dir, {k: v for k, v in shape.items() if k != "name"})
                    args = data if isinstance(data, (tuple, list)) else (data,)
                    expected = ref_kernel(*args)
                    received = custom_kernel(*args)
                reasons = verbose_allclose(received, expected, rtol=rtol, atol=atol)
                row["correct"] = "true" if not reasons else "false"
                row["max_abs_diff"] = f"{_max_abs_diff(received, expected):.6g}"
                if reasons:
                    row["error"] = " | ".join(reasons[:3])
            except Exception as exc:  # noqa: BLE001
                row["correct"] = "false"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    if mode in ("bench", "both"):
        for shape in task.get("shapes", {}).get("benchmarks", []) or []:
            row = _row(kernel, shape, "bench")
            try:
                set_seed(int(shape.get("seed", DEFAULT_SEED)))
                data = _load_inputs(kernel_dir, {k: v for k, v in shape.items() if k != "name"})
                args = data if isinstance(data, (tuple, list)) else (data,)
                mean_us, p50_us, min_us = _time_kernel(custom_kernel, args)
                row["mean_us"] = f"{mean_us:.3f}"
                row["p50_us"] = f"{p50_us:.3f}"
                row["min_us"] = f"{min_us:.3f}"
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    return rows


def _print_summary(rows: list[dict]) -> None:
    by_kernel: dict[str, list[dict]] = {}
    for r in rows:
        by_kernel.setdefault(r["kernel"], []).append(r)

    print("\n## Sweep summary\n")
    print("| kernel | shape | mode | min_us | p50_us | mean_us | correct | error |")
    print("|---|---|---|---:|---:|---:|---|---|")
    for kernel, krows in by_kernel.items():
        for r in krows:
            err = (r["error"] or "").replace("|", "\\|")[:60]
            print(
                f"| {kernel} | {r['shape']} | {r['mode']} | "
                f"{r['min_us'] or '-'} | {r['p50_us'] or '-'} | {r['mean_us'] or '-'} | "
                f"{r['correct'] or '-'} | {err} |"
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep AMD MI300X kernels into a single CSV")
    p.add_argument("--kernels", type=str, default=",".join(DEFAULT_KERNELS),
                   help="comma-separated kernel names (default: all four)")
    p.add_argument("--mode", choices=["test", "bench", "both"], default="both")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]

    print(f"PyTorch {torch.__version__} | ROCm/HIP {getattr(torch.version, 'hip', None)} | "
          f"GPU available: {torch.cuda.is_available()}")
    print(f"Sweeping kernels: {kernels} | mode: {args.mode}")

    all_rows: list[dict] = []
    for k in kernels:
        try:
            all_rows.extend(_run_kernel(k, args.mode, repo_root))
        except Exception as exc:  # noqa: BLE001 — keep sweep alive
            traceback.print_exc()
            all_rows.append({**_row(k, {}, args.mode), "error": f"sweep-aborted: {exc}"})

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"sweep_{ts}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in all_rows:
            w.writerow({c: row.get(c, "") for c in CSV_COLUMNS})
    print(f"\nWrote {len(all_rows)} rows to {out_csv}")

    _print_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
