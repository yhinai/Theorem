#!/usr/bin/env python3
"""Per-shape Triton config autotune for the four MI300X kernels.

For each (kernel, shape) we sweep a small grid of (block sizes, num_warps,
num_stages) by monkey-patching the kernel's SHAPE_CONFIGS dict and timing
custom_kernel(data) on the AMD device. Best config per shape is written to
results/autotune_summary.csv and printed as a markdown table.

Run on the AMD box (PyTorch ROCm + Triton 3.1+ required). No GPU on the
authoring machine — this script is build-validated by ast.parse only.

Usage:
    python benchmarks/autotune.py                       # all kernels, both modes
    python benchmarks/autotune.py --kernels causal_conv1d
    python benchmarks/autotune.py --quick               # 1/3 of the grid

Output:
    results/autotune_<kernel>.json  (full sweep, per shape)
    results/autotune_summary.csv    (best config per shape, all kernels)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml  # type: ignore

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------
# Per-kernel sweep grids and config-application strategies.
# ----------------------------------------------------------------------------

@dataclass
class GridSpec:
    name: str
    grid: dict[str, list]  # field -> candidate values
    apply_fn: Callable[[Any, dict, tuple], None]


def _filter_lds_budget(combo: dict, budget: int = 65536) -> bool:
    """Skip combos whose BLOCK_S * BLOCK_D blow the LDS budget guess."""
    bs = combo.get("BLOCK_S", 0)
    bd = combo.get("BLOCK_D", 0)
    if bs and bd and bs * bd > budget:
        return False
    return True


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        d = dict(zip(keys, combo))
        if _filter_lds_budget(d):
            out.append(d)
    return out


def apply_causal_conv1d(mod: Any, cfg: dict, shape_key: tuple) -> None:
    mod.SHAPE_CONFIGS[shape_key] = {
        "BLOCK_S": cfg["BLOCK_S"],
        "BLOCK_D": cfg["BLOCK_D"],
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }


def apply_chunk_fwd_h(mod: Any, cfg: dict, shape_key: tuple) -> None:
    mod.SHAPE_CONFIGS[shape_key] = {
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }


def apply_chunk_fwd_o(mod: Any, cfg: dict, shape_key: tuple) -> None:
    mod.SHAPE_CONFIGS[shape_key] = {
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }


def apply_recompute_w_u(mod: Any, cfg: dict, shape_key: tuple) -> None:
    new_cfg = {
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
    }
    if "GROUP_SIZE" in cfg:
        new_cfg["GROUP_SIZE"] = cfg["GROUP_SIZE"]
    mod.SHAPE_CONFIGS[shape_key] = new_cfg


GRIDS: dict[str, GridSpec] = {
    "causal_conv1d": GridSpec(
        name="causal_conv1d",
        grid={
            "BLOCK_S": [64, 128, 256, 512, 1024],
            "BLOCK_D": [16, 32, 64, 128],
            "num_warps": [4, 8, 16],
            "num_stages": [1, 2, 3],
        },
        apply_fn=apply_causal_conv1d,
    ),
    "chunk_fwd_h": GridSpec(
        name="chunk_fwd_h",
        grid={
            "num_warps": [4, 8, 16],
            "num_stages": [1, 2, 3, 4],
        },
        apply_fn=apply_chunk_fwd_h,
    ),
    "chunk_fwd_o": GridSpec(
        name="chunk_fwd_o",
        grid={
            "num_warps": [4, 8, 16],
            "num_stages": [1, 2, 3],
        },
        apply_fn=apply_chunk_fwd_o,
    ),
    "recompute_w_u": GridSpec(
        name="recompute_w_u",
        grid={
            "num_warps": [4, 8, 16],
            "num_stages": [1, 2, 3],
            "GROUP_SIZE": [4, 8, 16],
        },
        apply_fn=apply_recompute_w_u,
    ),
}


# ----------------------------------------------------------------------------
# Shape loading (read directly from kernels/<name>/task.yml)
# ----------------------------------------------------------------------------

def load_shapes(kernel: str, mode: str) -> list[dict]:
    task_path = ROOT / "kernels" / kernel / "task.yml"
    with open(task_path) as f:
        task = yaml.safe_load(f)
    shapes = []
    if mode in ("test", "both"):
        shapes.extend(task["shapes"]["tests"])
    if mode in ("bench", "both"):
        shapes.extend(task["shapes"]["benchmarks"])
    return shapes


def shape_key(kernel: str, shape: dict) -> tuple:
    """Map a shape dict to the tuple key the kernel uses in SHAPE_CONFIGS."""
    if kernel == "causal_conv1d":
        return (shape["B"], shape["D"], shape["S"], shape["W"])
    return (shape["B"], shape["T"], shape["H"], shape["K"], shape["V"])


def shape_label(shape: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items() if k != "name")


# ----------------------------------------------------------------------------
# Inputs and timing
# ----------------------------------------------------------------------------

def import_kernel_pkg(kernel: str):
    """Force a fresh import of kernels.<kernel> AND its .kernel submodule
    so monkey-patches to SHAPE_CONFIGS stick. Returns the submodule (where
    SHAPE_CONFIGS and the @triton.jit kernel function live).
    """
    pkg_name = f"kernels.{kernel}"
    sub_name = f"{pkg_name}.kernel"
    # remove cached submodules so re-import re-evaluates SHAPE_CONFIGS users
    for k in list(sys.modules.keys()):
        if k == pkg_name or k.startswith(pkg_name + "."):
            del sys.modules[k]
    importlib.import_module(pkg_name)
    sub = importlib.import_module(sub_name)
    return sub


def import_reference_module(kernel: str):
    """Import the kernel's reference.py (separate from the package import to
    avoid relative-import side effects when we re-import the kernel package)."""
    import importlib.util
    ref_path = ROOT / "kernels" / kernel / "reference.py"
    mod_name = f"_autotune_ref_{kernel}"
    spec = importlib.util.spec_from_file_location(mod_name, ref_path)
    assert spec and spec.loader, f"could not load reference.py for {kernel}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # required for @dataclass at exec time
    spec.loader.exec_module(mod)
    return mod


def make_inputs(ref_mod: Any, shape: dict):
    fn = getattr(ref_mod, "generate_input", None) or getattr(ref_mod, "generate_data", None)
    if fn is None:
        raise AttributeError(f"{ref_mod}: no generate_input/generate_data")
    return fn(**shape)


def time_kernel(custom_fn: Callable, data: Any, warmup: int = 5, iters: int = 25) -> float:
    """Return min wall time in microseconds across `iters` iterations."""
    for _ in range(warmup):
        custom_fn(data)
    torch.cuda.synchronize()

    times_us: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        custom_fn(data)
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return min(times_us)


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------

def sweep_kernel(kernel: str, mode: str, quick: bool) -> dict:
    grid_spec = GRIDS[kernel]
    grid = expand_grid(grid_spec.grid)
    if quick:
        grid = grid[::3]

    shapes = load_shapes(kernel, mode)
    print(f"\n=== {kernel} ===  configs={len(grid)}  shapes={len(shapes)}")
    out: dict[str, Any] = {"kernel": kernel, "configs_tested": len(grid), "by_shape": {}}

    ref_mod = import_reference_module(kernel)

    for shape in shapes:
        key = shape_key(kernel, shape)
        label = shape_label(shape)
        print(f"  shape: {label}")

        # Make inputs once per shape (deterministic via seed in shape).
        try:
            data = make_inputs(ref_mod, shape)
        except Exception as e:
            print(f"    INPUT GEN FAIL: {e}")
            out["by_shape"][label] = {"error": f"input-gen: {e}"}
            continue

        results = []
        baseline_us = None
        # First time the original config to get baseline
        try:
            mod = import_kernel_pkg(kernel)
            # Re-import to pick up the original SHAPE_CONFIGS in kernel.py
            current_cfg = mod.SHAPE_CONFIGS.get(key)
            t = time_kernel(mod.custom_kernel, data)
            baseline_us = t
            results.append({"config": current_cfg, "min_us": t, "is_baseline": True})
        except Exception as e:
            print(f"    BASELINE FAIL: {e}")

        # Sweep
        for i, cfg in enumerate(grid):
            try:
                mod = import_kernel_pkg(kernel)
                grid_spec.apply_fn(mod, cfg, key)
                t = time_kernel(mod.custom_kernel, data)
                results.append({"config": cfg, "min_us": t})
            except Exception as e:
                # capture only the first 200 chars
                results.append({"config": cfg, "error": str(e)[:200]})

        # Sort by min_us asc (errors at the bottom)
        ok = [r for r in results if "min_us" in r]
        bad = [r for r in results if "error" in r]
        ok.sort(key=lambda r: r["min_us"])
        out["by_shape"][label] = {
            "shape_key": list(key),
            "baseline_us": baseline_us,
            "best": ok[0] if ok else None,
            "all_ok": ok[:10],   # top 10 only to keep file small
            "n_failed": len(bad),
            "first_failure": bad[0] if bad else None,
        }

        if ok:
            best = ok[0]
            improvement = (
                f"{(baseline_us - best['min_us']) / baseline_us * 100:+.1f}%"
                if baseline_us else "n/a"
            )
            print(f"    baseline={baseline_us:.2f}us  best={best['min_us']:.2f}us ({improvement})")
            print(f"    best cfg: {best['config']}")
        else:
            print(f"    NO PASSING CONFIGS")

    return out


def write_summary(all_results: list[dict], out_csv: Path) -> list[dict]:
    rows = []
    for kr in all_results:
        for label, info in kr["by_shape"].items():
            if not info.get("best"):
                rows.append({
                    "kernel": kr["kernel"],
                    "shape": label,
                    "baseline_us": info.get("baseline_us"),
                    "best_us": None,
                    "best_config": None,
                    "improvement_pct": None,
                    "n_failed": info.get("n_failed", 0),
                    "error": info.get("first_failure", {}).get("error", "")[:120] if info.get("first_failure") else "",
                })
                continue
            base = info["baseline_us"] or float("nan")
            best = info["best"]["min_us"]
            improvement = (base - best) / base * 100 if base else None
            rows.append({
                "kernel": kr["kernel"],
                "shape": label,
                "baseline_us": f"{base:.3f}" if base else "",
                "best_us": f"{best:.3f}",
                "best_config": json.dumps(info["best"]["config"]),
                "improvement_pct": f"{improvement:+.2f}" if improvement is not None else "",
                "n_failed": info["n_failed"],
                "error": "",
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kernel", "shape", "baseline_us", "best_us", "best_config", "improvement_pct", "n_failed", "error"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nwrote {out_csv}")
    return rows


def print_md_table(rows: list[dict]) -> None:
    print("\n## Autotune summary\n")
    print("| kernel | shape | baseline_us | best_us | improvement | best_config |")
    print("|---|---|---:|---:|---:|---|")
    for r in rows:
        cfg = (r["best_config"] or "")[:80]
        print(f"| {r['kernel']} | {r['shape']} | {r['baseline_us']} | {r['best_us']} | {r['improvement_pct']} | `{cfg}` |")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernels", default="all", help="comma-separated subset; 'all' = all 4")
    p.add_argument("--mode", choices=["test", "bench", "both"], default="bench")
    p.add_argument("--quick", action="store_true", help="sweep 1/3 of the grid")
    args = p.parse_args(argv)

    selected = list(GRIDS.keys()) if args.kernels == "all" else args.kernels.split(",")
    print(f"PyTorch {torch.__version__} | ROCm/HIP {torch.version.hip} | GPU available: {torch.cuda.is_available()}")
    print(f"Sweep: kernels={selected} mode={args.mode} quick={args.quick}")

    t0 = time.time()
    all_results = []
    for kernel in selected:
        if kernel not in GRIDS:
            print(f"skipping unknown kernel: {kernel}")
            continue
        try:
            result = sweep_kernel(kernel, args.mode, args.quick)
            all_results.append(result)
            # write per-kernel JSON
            jpath = ROOT / "results" / f"autotune_{kernel}.json"
            jpath.parent.mkdir(parents=True, exist_ok=True)
            with open(jpath, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  wrote {jpath}")
        except Exception as e:
            print(f"  KERNEL FAIL: {e}")
            traceback.print_exc()

    rows = write_summary(all_results, ROOT / "results" / "autotune_summary.csv")
    print_md_table(rows)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
