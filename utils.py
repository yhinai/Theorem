"""Utility helpers for the AMD MI300X kernel test/benchmark harness.

All helpers are intentionally lightweight and have no hard dependency on the
kernel directories — those are imported lazily so this module can be loaded
even before the kernel sources exist on disk.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Pin every RNG we care about for reproducible runs on the AMD device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # The torch namespace is `torch.cuda` even on ROCm — vestigial naming.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verbose_allclose(
    received,
    expected,
    rtol: float,
    atol: float,
    max_print: int = 5,
) -> list[str]:
    """Return a list of human-readable mismatch reasons. Empty list = pass.

    Tuples/lists of tensors are compared element-wise.
    """
    if isinstance(received, (tuple, list)) or isinstance(expected, (tuple, list)):
        if type(received) is not type(expected) or len(received) != len(expected):
            return [f"container mismatch: received={type(received).__name__}({len(received) if hasattr(received, '__len__') else '?'}), "
                    f"expected={type(expected).__name__}({len(expected) if hasattr(expected, '__len__') else '?'})"]
        out: list[str] = []
        for i, (r, e) in enumerate(zip(received, expected)):
            for reason in verbose_allclose(r, e, rtol=rtol, atol=atol, max_print=max_print):
                out.append(f"[{i}] {reason}")
        return out

    reasons: list[str] = []
    if not isinstance(received, torch.Tensor) or not isinstance(expected, torch.Tensor):
        reasons.append(f"type mismatch: received={type(received)}, expected={type(expected)}")
        return reasons
    if received.shape != expected.shape:
        reasons.append(f"shape mismatch: received={tuple(received.shape)}, expected={tuple(expected.shape)}")
        return reasons
    if received.dtype != expected.dtype:
        reasons.append(f"dtype mismatch: received={received.dtype}, expected={expected.dtype}")

    r = received.detach().to(torch.float32).cpu()
    e = expected.detach().to(torch.float32).cpu()

    nan_mismatch = torch.isnan(r) ^ torch.isnan(e)
    inf_mismatch = torch.isinf(r) ^ torch.isinf(e)
    if nan_mismatch.any():
        reasons.append(f"NaN pattern mismatch in {int(nan_mismatch.sum())} elements")
    if inf_mismatch.any():
        reasons.append(f"Inf pattern mismatch in {int(inf_mismatch.sum())} elements")

    finite_mask = torch.isfinite(r) & torch.isfinite(e)
    diff = torch.where(finite_mask, (r - e).abs(), torch.zeros_like(r))
    tol = atol + rtol * e.abs()
    bad = finite_mask & (diff > tol)
    n_bad = int(bad.sum())
    if n_bad > 0:
        max_abs = float(diff.max())
        reasons.append(f"{n_bad} elements exceed tolerance (rtol={rtol}, atol={atol}); max|diff|={max_abs:.3e}")
        idxs = torch.nonzero(bad, as_tuple=False)[:max_print]
        for idx in idxs:
            t = tuple(int(i) for i in idx)
            reasons.append(f"  at {t}: received={float(r[t]):.6g}, expected={float(e[t]):.6g}")
    return reasons


def make_match_reference(
    ref_fn: Callable[..., torch.Tensor],
    rtol: float,
    atol: float,
) -> Callable[[Any, torch.Tensor], tuple[bool, str]]:
    """Build a checker that compares an output to ref_fn(data)."""
    def _check(data: Any, output: torch.Tensor) -> tuple[bool, str]:
        expected = ref_fn(*data) if isinstance(data, (tuple, list)) else ref_fn(data)
        reasons = verbose_allclose(output, expected, rtol=rtol, atol=atol)
        return (len(reasons) == 0, "; ".join(reasons))
    return _check


@contextmanager
def DeterministicContext():
    """Pin determinism flags. Device-agnostic via torch.use_deterministic_algorithms."""
    try:
        prev = torch.are_deterministic_algorithms_enabled()
    except Exception:
        prev = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            torch.use_deterministic_algorithms(prev, warn_only=True)
        except Exception:
            pass


def load_task(kernel_dir: Path) -> dict:
    """Read task.yml via PyYAML and return the parsed dict."""
    import yaml  # local import: keeps top-level import surface minimal

    path = Path(kernel_dir) / "task.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def import_kernel(kernel_dir: Path) -> tuple[Callable, Callable]:
    """Import (custom_kernel, ref_kernel) from kernels/<name>/.

    The kernel modules contain relative imports (``from .kernel import ...``),
    so we resolve them as a real package. The ``kernels/`` parent is added
    to ``sys.path`` (idempotent) and the module is imported by dotted name.
    """
    import importlib

    kernel_dir = Path(kernel_dir).resolve()
    init_file = kernel_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"missing __init__.py at {init_file}")

    project_root = kernel_dir.parent.parent  # parent of "kernels/"
    sys_path_entry = str(project_root)
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)

    dotted = f"kernels.{kernel_dir.name}"
    module = importlib.import_module(dotted)

    custom = getattr(module, "custom_kernel", None)
    ref = getattr(module, "ref_kernel", None)
    if custom is None or ref is None:
        raise AttributeError(f"{init_file} must export `custom_kernel` and `ref_kernel`")
    return custom, ref
