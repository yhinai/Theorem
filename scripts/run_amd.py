#!/usr/bin/env python3
"""Quick-run smoke test for the four AMD MI300X kernels.

Imports each kernel module from `kernels/<name>/`, generates the smallest
test shape per `task.yml`, runs `ref_kernel` and `custom_kernel`, and
checks agreement with `torch.allclose`. Prints env info up front and
never crashes on a single kernel's failure -- it logs and continues.

Usage:
    python scripts/run_amd.py
    python scripts/run_amd.py --only causal_conv1d,chunk_fwd_h
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import traceback
from pathlib import Path

# Make `from kernels.<name> import ...` work when run from anywhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

ALL_KERNELS = ("causal_conv1d", "chunk_fwd_h", "chunk_fwd_o", "recompute_w_u")

# Smallest test shape per kernel (matches kernels/<name>/task.yml first entry).
SMALLEST_SHAPES = {
    "causal_conv1d":  {"B": 1, "D": 64, "S": 64, "W": 4,  "seed": 4242},
    "chunk_fwd_h":    {"B": 1, "T": 64, "H": 1, "K": 64, "V": 64, "seed": 4242},
    "chunk_fwd_o":    {"B": 1, "T": 64, "H": 1, "K": 64, "V": 64, "seed": 4242},
    "recompute_w_u":  {"B": 1, "T": 64, "H": 2, "K": 64, "V": 64, "seed": 4242},
}

TOL = {"rtol": 1e-2, "atol": 1e-2}


def print_env() -> None:
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"Python : {platform.python_version()}  ({sys.executable})")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        hip = getattr(torch.version, "hip", None)
        print(f"HIP    : {hip if hip else '(no HIP build)'}")
        if torch.cuda.is_available():
            print(f"GPU    : {torch.cuda.get_device_name(0)}")
        else:
            print("GPU    : (no GPU detected)")
    except Exception as e:
        print(f"PyTorch: import failed ({e})")
    try:
        import triton
        print(f"Triton : {triton.__version__}")
    except Exception as e:
        print(f"Triton : import failed ({e})")
    print()


def _build_data(name: str, shape: dict):
    """Return (data, ref_fn, custom_fn) for a kernel by name."""
    mod = importlib.import_module(f"kernels.{name}")

    if name == "causal_conv1d":
        from kernels.causal_conv1d.reference import generate_input
        data = generate_input(**shape)  # (x, weight, bias) tuple
    elif name == "chunk_fwd_h":
        from kernels.chunk_fwd_h.reference import generate_input
        data = generate_input(**shape)  # dict
    elif name == "chunk_fwd_o":
        from kernels.chunk_fwd_o.reference import _build_inputs
        data = _build_inputs(device="cuda", **shape)  # dict
    elif name == "recompute_w_u":
        from kernels.recompute_w_u.reference import generate_data
        data = generate_data(device="cuda", **shape)  # Data dataclass
    else:
        raise ValueError(f"unknown kernel: {name}")

    return data, mod.ref_kernel, mod.custom_kernel


def _allclose(ref, got) -> bool:
    import torch
    if isinstance(ref, (tuple, list)):
        if not isinstance(got, (tuple, list)) or len(ref) != len(got):
            return False
        return all(torch.allclose(r, g, **TOL) for r, g in zip(ref, got))
    return torch.allclose(ref, got, **TOL)


def run_one(name: str) -> bool:
    shape = SMALLEST_SHAPES[name]
    print(f"--- {name} (shape={shape}) ---")
    try:
        data, ref_fn, custom_fn = _build_data(name, shape)
    except Exception as e:
        print(f"  IMPORT/GEN FAIL: {e}")
        traceback.print_exc(limit=2)
        return False

    try:
        ref = ref_fn(data)
    except Exception as e:
        print(f"  REF FAIL: {e}")
        traceback.print_exc(limit=2)
        return False

    try:
        got = custom_fn(data)
    except Exception as e:
        print(f"  CUSTOM FAIL: {e}")
        traceback.print_exc(limit=2)
        return False

    try:
        ok = _allclose(ref, got)
    except Exception as e:
        print(f"  COMPARE FAIL: {e}")
        return False

    print(f"  {'PASS' if ok else 'FAIL'}  (rtol={TOL['rtol']}, atol={TOL['atol']})")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        type=str,
        default="",
        help=f"comma-separated subset of {','.join(ALL_KERNELS)}",
    )
    args = p.parse_args()

    print_env()

    selected = (
        [s.strip() for s in args.only.split(",") if s.strip()]
        if args.only
        else list(ALL_KERNELS)
    )
    bad = [s for s in selected if s not in ALL_KERNELS]
    if bad:
        print(f"ERROR: unknown kernel(s): {bad}; valid: {ALL_KERNELS}")
        return 2

    results = {name: run_one(name) for name in selected}

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:18s}  {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
