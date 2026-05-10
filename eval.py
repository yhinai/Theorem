"""Per-kernel test + benchmark runner for the AMD MI300X harness.

Usage:
    python eval.py test kernels/causal_conv1d/
    python eval.py benchmark kernels/chunk_fwd_h/
    python eval.py both kernels/recompute_w_u/

`test` runs correctness checks vs. ref_kernel; `benchmark` measures
custom_kernel latency on the AMD device using GPU events.
"""
from __future__ import annotations

import argparse
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

WARMUP_ITERS = 5
TIMED_ITERS = 50
DEFAULT_SEED = 0


def _print_header() -> None:
    """Print a banner with PyTorch / Triton / ROCm / GPU info."""
    print("=" * 72)
    print(f"PyTorch:      {torch.__version__}")
    triton_ver = "not installed"
    try:
        spec = importlib.util.find_spec("triton")
        if spec is not None:
            import triton  # type: ignore
            triton_ver = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        triton_ver = f"import-error: {exc!r}"
    print(f"Triton:       {triton_ver}")

    rocm_ver = getattr(torch.version, "hip", None)
    print(f"ROCm/HIP:     {rocm_ver or 'n/a'}")

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        print(f"GPU:          {name}")
    else:
        print("GPU:          NOT AVAILABLE — running on host fallback")
    print("=" * 72)


def _load_inputs(kernel_dir: Path, shape_kwargs: dict) -> Any:
    """Import reference.py from the kernel dir and call generate_input(**shape_kwargs)."""
    ref_path = Path(kernel_dir).resolve() / "reference.py"
    if not ref_path.exists():
        raise FileNotFoundError(f"missing reference.py at {ref_path}")
    spec = importlib.util.spec_from_file_location(f"_ref_{Path(kernel_dir).name}", ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {ref_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen = getattr(mod, "generate_input", None)
    if gen is None:
        raise AttributeError(f"{ref_path} must export generate_input(**kwargs)")
    return gen(**shape_kwargs)


def _shape_label(shape: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in shape.items() if k != "name")


def run_tests(kernel_dir: Path) -> tuple[int, int]:
    """Return (passed, total)."""
    task = load_task(kernel_dir)
    tests = task.get("shapes", {}).get("tests", []) or []
    tols = task.get("tolerances", {}) or {}
    rtol = float(tols.get("rtol", 1e-3))
    atol = float(tols.get("atol", 1e-3))

    custom_kernel, ref_kernel = import_kernel(kernel_dir)
    passed = 0
    total = len(tests)
    print(f"\n[test] {kernel_dir}  ({total} shapes; rtol={rtol}, atol={atol})")
    for shape in tests:
        label = _shape_label(shape)
        try:
            set_seed(int(shape.get("seed", DEFAULT_SEED)))
            with DeterministicContext():
                data = _load_inputs(kernel_dir, {k: v for k, v in shape.items() if k != "name"})
                args = data if isinstance(data, (tuple, list)) else (data,)
                expected = ref_kernel(*args)
                received = custom_kernel(*args)
            reasons = verbose_allclose(received, expected, rtol=rtol, atol=atol)
            if not reasons:
                print(f"  PASS  {label}")
                passed += 1
            else:
                print(f"  FAIL  {label}")
                for r in reasons:
                    print(f"        {r}")
        except Exception as exc:  # noqa: BLE001 — surface every failure mode
            print(f"  ERROR {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2, file=sys.stdout)
    print(f"[test] {passed}/{total} passing")
    return passed, total


def _time_kernel(custom_kernel, args) -> tuple[float, float, float]:
    """Run warmup + timed iters; return (mean_us, p50_us, min_us)."""
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
    times_us = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(TIMED_ITERS)]
    times_us.sort()
    mean_us = sum(times_us) / len(times_us)
    p50_us = times_us[len(times_us) // 2]
    min_us = times_us[0]
    return mean_us, p50_us, min_us


def run_benchmarks(kernel_dir: Path) -> None:
    task = load_task(kernel_dir)
    benches = task.get("shapes", {}).get("benchmarks", []) or []
    custom_kernel, _ref_kernel = import_kernel(kernel_dir)

    print(f"\n[bench] {kernel_dir}  ({len(benches)} shapes; warmup={WARMUP_ITERS}, iters={TIMED_ITERS})")
    print(f"  {'shape':<48} {'mean_us':>10} {'p50_us':>10} {'min_us':>10}")
    for shape in benches:
        label = _shape_label(shape)
        try:
            set_seed(int(shape.get("seed", DEFAULT_SEED)))
            data = _load_inputs(kernel_dir, {k: v for k, v in shape.items() if k != "name"})
            args = data if isinstance(data, (tuple, list)) else (data,)
            mean_us, p50_us, min_us = _time_kernel(custom_kernel, args)
            print(f"  {label:<48} {mean_us:>10.2f} {p50_us:>10.2f} {min_us:>10.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<48} ERROR: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AMD MI300X kernel eval harness")
    p.add_argument("mode", choices=["test", "benchmark", "both"])
    p.add_argument("kernel_dir", type=Path)
    args = p.parse_args(argv)

    _print_header()
    kernel_dir = args.kernel_dir
    if not kernel_dir.exists():
        print(f"ERROR: kernel directory {kernel_dir} does not exist")
        return 2

    try:
        if args.mode in ("test", "both"):
            run_tests(kernel_dir)
        if args.mode in ("benchmark", "both"):
            run_benchmarks(kernel_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
