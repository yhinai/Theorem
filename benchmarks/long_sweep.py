#!/usr/bin/env python3
"""Long-running random + hill-climb sweep for the MI300X kernels.

Designed to be left running for hours. Picks random (kernel, shape, config)
triples (or hill-climbs from the current best), times each, and appends one
JSON line per sample to a JSONL log. Keeps an in-memory running best per
(kernel, shape) and prints a progress summary every minute.

CLI:
    python benchmarks/long_sweep.py --hours 6
    python benchmarks/long_sweep.py --hours 0.5 --strategy random
    python benchmarks/long_sweep.py --hours 2 --strategy hill
    python benchmarks/long_sweep.py --hours 8 --strategy mixed  (default)

Output:
    results/long_sweep_<ts>.jsonl          every sample (one JSON object per line)
    results/long_sweep_<ts>.summary.csv    all-time best per (kernel, shape)

Signals:
    SIGINT / SIGTERM  →  flush summary CSV, exit cleanly
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.autotune import (  # type: ignore
    GRIDS, load_shapes, shape_key, shape_label,
    import_kernel_pkg, import_reference_module, make_inputs,
)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

def time_kernel(custom_fn, data, warmup: int = 5, iters: int = 25) -> float:
    """Min-of-iters timing in microseconds."""
    for _ in range(warmup):
        custom_fn(data)
    torch.cuda.synchronize()
    min_us = float("inf")
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        custom_fn(data)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000.0
        if us < min_us:
            min_us = us
    return min_us


# ----------------------------------------------------------------------------
# Config search strategies
# ----------------------------------------------------------------------------

def random_config(grid: dict[str, list]) -> dict:
    return {k: random.choice(v) for k, v in grid.items()}


def perturb_config(cfg: dict, grid: dict[str, list], n_changes: int = 1) -> dict:
    """Hill-climb neighbor: change `n_changes` random knobs by one grid step."""
    new = dict(cfg)
    keys = list(grid.keys())
    if not keys:
        return new
    for _ in range(n_changes):
        k = random.choice(keys)
        candidates = grid[k]
        try:
            i = candidates.index(cfg.get(k))
            step = random.choice([-1, 1])
            new[k] = candidates[max(0, min(len(candidates) - 1, i + step))]
        except (ValueError, KeyError):
            new[k] = random.choice(candidates)
    return new


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

def write_summary(out_path: Path, best_per: dict, n_samples: int, n_failures: int) -> None:
    summary_path = out_path.with_suffix(".summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "shape", "best_us", "config", "found_at"])
        for (kernel, label), info in sorted(best_per.items()):
            w.writerow([
                kernel, label, f"{info['us']:.3f}",
                json.dumps(info["cfg"]),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["ts"])),
            ])
    print(f"\nwrote {summary_path}")
    print(f"\nALL-TIME BEST per (kernel, shape)  [samples={n_samples}, failures={n_failures}]:")
    for (k, s), info in sorted(best_per.items()):
        cfg = json.dumps(info["cfg"])[:90]
        print(f"  {k:18s} {s:48s} {info['us']:8.2f} us  {cfg}")


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=1.0,
                   help="how long to run, in hours (decimal allowed)")
    p.add_argument("--strategy", choices=["random", "hill", "mixed"], default="mixed",
                   help="random | hill (climb from best) | mixed (70%% random, 30%% hill)")
    p.add_argument("--kernels", default="all",
                   help="comma-separated subset (default: all 4)")
    p.add_argument("--out", default=None,
                   help="JSONL output path; default = results/long_sweep_<timestamp>.jsonl")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print-every", type=float, default=60.0,
                   help="seconds between progress summaries")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else ROOT / "results" / f"long_sweep_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kernels_arg = args.kernels
    kernels = list(GRIDS.keys()) if kernels_arg == "all" else kernels_arg.split(",")
    kernels = [k for k in kernels if k in GRIDS]
    if not kernels:
        print("no valid kernels selected", file=sys.stderr)
        return 1

    print(f"PyTorch {torch.__version__} | ROCm/HIP {torch.version.hip} | "
          f"GPU available: {torch.cuda.is_available()}")
    print(f"Long sweep | hours={args.hours} strategy={args.strategy} kernels={kernels}")
    print(f"Output: {out_path}")
    print()

    # Cache reference modules per kernel; inputs cached lazily per (kernel, shape).
    ref_mods = {k: import_reference_module(k) for k in kernels}
    inputs_cache: dict[tuple, Any] = {}

    best_per: dict[tuple, dict] = {}  # (kernel, shape_label) -> {us, cfg, ts}
    start_time = time.time()
    end_time = start_time + args.hours * 3600
    last_print = start_time

    n_samples = 0
    n_failures = 0

    # Open output JSONL with line buffering so each sample is durable on disk.
    out_fh = open(out_path, "a", buffering=1)

    def on_signal(signum, frame):
        print(f"\n\n[!] caught signal {signum} — flushing summary", file=sys.stderr)
        out_fh.flush()
        out_fh.close()
        write_summary(out_path, best_per, n_samples, n_failures)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        while time.time() < end_time:
            kernel = random.choice(kernels)
            grid_spec = GRIDS[kernel]
            shapes = load_shapes(kernel, "bench")
            shape = random.choice(shapes)
            key = shape_key(kernel, shape)
            label = shape_label(shape)
            cache_key = (kernel, key)

            # Get / cache inputs for this shape.
            if cache_key not in inputs_cache:
                try:
                    inputs_cache[cache_key] = make_inputs(ref_mods[kernel], shape)
                except Exception as ex:
                    n_failures += 1
                    continue
            data = inputs_cache[cache_key]

            # Pick the next config.
            cur_best = best_per.get((kernel, label))
            if cur_best is None or args.strategy == "random":
                cfg = random_config(grid_spec.grid)
            elif args.strategy == "hill":
                cfg = perturb_config(cur_best["cfg"], grid_spec.grid)
            else:  # mixed
                if random.random() < 0.7:
                    cfg = random_config(grid_spec.grid)
                else:
                    cfg = perturb_config(cur_best["cfg"], grid_spec.grid)

            # Time it (catching any failure cleanly).
            t = float("nan")
            status = "ok"
            error = ""
            try:
                mod = import_kernel_pkg(kernel)
                grid_spec.apply_fn(mod, cfg, key)
                t = time_kernel(mod.custom_kernel, data)
            except Exception as ex:
                status = "fail"
                error = f"{type(ex).__name__}: {str(ex)[:160]}"
                n_failures += 1

            # Log one JSON line per sample.
            sample = {
                "ts": round(time.time(), 3),
                "kernel": kernel,
                "shape": label,
                "config": cfg,
                "min_us": None if t != t else round(t, 3),  # None if NaN
                "status": status,
            }
            if error:
                sample["error"] = error
            out_fh.write(json.dumps(sample) + "\n")
            n_samples += 1

            # Update running best.
            if status == "ok":
                key2 = (kernel, label)
                prev = best_per.get(key2)
                if prev is None or t < prev["us"]:
                    delta = "" if prev is None else f" (was {prev['us']:.2f}, -{(prev['us']-t)/prev['us']*100:.1f}%)"
                    best_per[key2] = {"us": t, "cfg": cfg, "ts": sample["ts"]}
                    print(f"  NEW BEST  {kernel:18s} {label:48s} {t:8.2f} us  {cfg}{delta}")

            # Periodic progress.
            now = time.time()
            if now - last_print >= args.print_every:
                elapsed = now - start_time
                remaining = end_time - now
                rate = n_samples / max(elapsed, 1.0)
                print(f"\n[{elapsed/60:.1f}min elapsed · {remaining/60:.1f}min left · "
                      f"{n_samples} samples · {rate:.1f} samples/s · {n_failures} fails]")
                for (k, s), info in sorted(best_per.items()):
                    print(f"   {k:18s} {s:48s} {info['us']:8.2f} us")
                print()
                last_print = now
    finally:
        out_fh.flush()
        out_fh.close()
        write_summary(out_path, best_per, n_samples, n_failures)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
