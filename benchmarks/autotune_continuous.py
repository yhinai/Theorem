#!/usr/bin/env python3
"""Continuous-loop autotuner with hill climbing + random restarts.

For each (kernel, shape) it:

1. Times the current SHAPE_CONFIGS entry to get a baseline.
2. Hill-climbs: try every "neighbor" config (one knob different by one step
   along the grid axis); if any neighbor is >= 1% faster, jump to it. Repeat
   until no neighbor improves -> local minimum.
3. Random-restart: pick a fresh random point in the grid and climb again.
4. Iterate steps 2-3 for `--restarts` rounds (default: 3).

Across all hill-climbs and all restarts, the all-time fastest config per
(kernel, shape) is the global-best estimate. Median-of-N timings smooths
measurement noise (--medians, default 3).

Outputs:
  results/autotune_continuous_<run-id>.csv     incremental best-so-far per iter
  results/autotune_continuous_summary.csv      final best per (kernel, shape)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml  # type: ignore

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Reuse pieces from the standard autotune module.
from benchmarks.autotune import (  # type: ignore
    GRIDS, GridSpec, expand_grid, load_shapes, shape_key, shape_label,
    import_kernel_pkg, import_reference_module, make_inputs,
)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def time_kernel(custom_fn: Callable, data: Any,
                warmup: int = 5, iters: int = 25, medians: int = 3) -> float:
    """Run `medians` independent measurements; return the MEDIAN of mins (us)."""
    samples = []
    for _ in range(medians):
        for _ in range(warmup):
            custom_fn(data)
        torch.cuda.synchronize()
        run_min = float("inf")
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            custom_fn(data)
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000.0
            if us < run_min:
                run_min = us
        samples.append(run_min)
    return statistics.median(samples)


def _safe_time(grid_spec: GridSpec, kernel: str, key: tuple, cfg: dict,
               data: Any, medians: int) -> float | None:
    try:
        mod = import_kernel_pkg(kernel)
        grid_spec.apply_fn(mod, cfg, key)
        return time_kernel(mod.custom_kernel, data, medians=medians)
    except Exception as e:
        print(f"      cfg {cfg} -> error: {str(e)[:120]}")
        return None


# --------------------------------------------------------------------------
# Hill climb: neighbors = one knob varied by one step on the grid axis
# --------------------------------------------------------------------------

def neighbors(cfg: dict, grid: dict[str, list]) -> list[dict]:
    out: list[dict] = []
    for k, candidates in grid.items():
        if k not in cfg:
            continue
        try:
            i = candidates.index(cfg[k])
        except ValueError:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(candidates):
                nb = dict(cfg)
                nb[k] = candidates[j]
                out.append(nb)
    return out


def hill_climb(grid_spec: GridSpec, kernel: str, key: tuple, start: dict,
               data: Any, medians: int, max_steps: int = 30
               ) -> tuple[dict, float, list[tuple[int, dict, float]]]:
    """Greedy hill climb. Returns (best_cfg, best_time_us, trace)."""
    cur = dict(start)
    cur_t = _safe_time(grid_spec, kernel, key, cur, data, medians)
    if cur_t is None:
        cur_t = float("inf")
    trace: list[tuple[int, dict, float]] = [(0, cur, cur_t)]

    for step in range(1, max_steps + 1):
        nbs = neighbors(cur, grid_spec.grid)
        random.shuffle(nbs)
        improved = False
        for nb in nbs:
            t = _safe_time(grid_spec, kernel, key, nb, data, medians)
            if t is not None and t < cur_t * 0.99:  # >=1% faster to jump
                cur, cur_t = nb, t
                trace.append((step, cur, cur_t))
                improved = True
                break
        if not improved:
            break  # local minimum
    return cur, cur_t, trace


def random_config(grid: dict[str, list]) -> dict:
    return {k: random.choice(v) for k, v in grid.items()}


# --------------------------------------------------------------------------
# Per-kernel sweep
# --------------------------------------------------------------------------

def sweep_kernel(kernel: str, restarts: int, medians: int,
                 mode: str, run_id: str) -> dict:
    grid_spec = GRIDS[kernel]
    shapes = load_shapes(kernel, mode)
    print(f"\n=== {kernel} ===  shapes={len(shapes)}  restarts={restarts}")
    out: dict[str, Any] = {"kernel": kernel, "shapes": {}}

    ref_mod = import_reference_module(kernel)
    csv_path = ROOT / "results" / f"autotune_continuous_{run_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for shape in shapes:
        key = shape_key(kernel, shape)
        label = shape_label(shape)
        print(f"\n  shape: {label}")

        try:
            data = make_inputs(ref_mod, shape)
        except Exception as e:
            print(f"    INPUT GEN FAIL: {e}")
            continue

        # Round 0: start from the kernel's existing SHAPE_CONFIGS entry.
        try:
            mod = import_kernel_pkg(kernel)
            current = dict(mod.SHAPE_CONFIGS.get(key) or {})
        except Exception:
            current = {}
        if not current:
            current = random_config(grid_spec.grid)

        # Pad with grid defaults if any required knob is missing.
        for k, v in grid_spec.grid.items():
            if k not in current:
                current[k] = v[0]

        all_time_best = current
        all_time_best_t = float("inf")

        for r in range(restarts + 1):
            if r == 0:
                start = current  # start from kernel default
                tag = "kernel-default"
            else:
                start = random_config(grid_spec.grid)
                tag = f"random-restart-{r}"
            print(f"    [{tag}] start={start}")
            best, best_t, trace = hill_climb(grid_spec, kernel, key, start,
                                             data, medians)
            print(f"    [{tag}] -> best={best}  ({best_t:.2f} us, {len(trace)} steps)")

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if csv_path.stat().st_size == 0:
                    w.writerow(["kernel", "shape", "restart", "step", "config", "us"])
                for step, cfg, t in trace:
                    w.writerow([kernel, label, tag, step, json.dumps(cfg), f"{t:.3f}"])

            if best_t < all_time_best_t:
                all_time_best, all_time_best_t = best, best_t

        out["shapes"][label] = {
            "shape_key": list(key),
            "best_config": all_time_best,
            "best_us": all_time_best_t,
        }
        print(f"  ALL-TIME BEST for {label}: {all_time_best_t:.2f} us  cfg={all_time_best}")

    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernels", default="all",
                   help="comma-separated subset; 'all' = all 4")
    p.add_argument("--mode", choices=["test", "bench", "both"], default="bench")
    p.add_argument("--restarts", type=int, default=3,
                   help="random restarts after the first hill-climb")
    p.add_argument("--medians", type=int, default=3,
                   help="median-of-N timings to smooth measurement noise")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    selected = list(GRIDS.keys()) if args.kernels == "all" else args.kernels.split(",")

    print(f"PyTorch {torch.__version__} | ROCm/HIP {torch.version.hip} "
          f"| GPU available: {torch.cuda.is_available()}")
    print(f"Continuous autotune | kernels={selected} mode={args.mode} "
          f"restarts={args.restarts} medians={args.medians}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    summary: list[dict] = []

    for kernel in selected:
        if kernel not in GRIDS:
            print(f"skipping unknown kernel: {kernel}")
            continue
        try:
            result = sweep_kernel(kernel, args.restarts, args.medians,
                                  args.mode, run_id)
            for label, info in result["shapes"].items():
                summary.append({
                    "kernel": kernel,
                    "shape": label,
                    "best_us": info["best_us"],
                    "best_config": json.dumps(info["best_config"]),
                })
        except Exception as e:
            print(f"  KERNEL FAIL: {e}")

    # Final summary CSV
    sum_path = ROOT / "results" / "autotune_continuous_summary.csv"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "shape", "best_us", "best_config"])
        w.writeheader()
        for row in summary:
            row["best_us"] = f"{row['best_us']:.3f}"
            w.writerow(row)
    print(f"\nwrote {sum_path}")

    # Markdown table
    print("\n## Continuous-autotune summary\n")
    print("| kernel | shape | best_us | best_config |")
    print("|---|---|---:|---|")
    for row in summary:
        cfg = row["best_config"][:80]
        print(f"| {row['kernel']} | {row['shape']} | {row['best_us']} | `{cfg}` |")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
