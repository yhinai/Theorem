#!/usr/bin/env python3
"""Read results/autotune_summary.csv and patch each kernel's SHAPE_CONFIGS
with the best config found per shape.

Idempotent: re-running with the same CSV produces the same kernel.py.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUMMARY = ROOT / "results" / "autotune_summary.csv"


def load_best_per_kernel() -> dict[str, list[tuple[tuple, dict]]]:
    """Return {kernel_name: [(shape_key_tuple, best_cfg_dict), ...]}."""
    if not SUMMARY.exists():
        print(f"FATAL: {SUMMARY} not found — run autotune first", file=sys.stderr)
        sys.exit(2)

    out: dict[str, list[tuple[tuple, dict]]] = {}
    with open(SUMMARY) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("best_config"):
                continue
            kernel = row["kernel"]
            cfg = json.loads(row["best_config"])
            # parse shape label like "B=1, T=64, H=1, K=64, V=64, seed=4242"
            kvs = {}
            for part in row["shape"].split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        kvs[k.strip()] = int(v.strip())
                    except ValueError:
                        pass
            if kernel == "causal_conv1d":
                key = (kvs["B"], kvs["D"], kvs["S"], kvs["W"])
            else:
                key = (kvs["B"], kvs["T"], kvs["H"], kvs["K"], kvs["V"])
            out.setdefault(kernel, []).append((key, cfg))
    return out


def render_dict_literal(d: dict) -> str:
    """Render a config dict as a python literal in a stable key order."""
    order = ["BLOCK_S", "BLOCK_D", "BLOCK_K", "BLOCK_V", "GROUP_SIZE", "num_warps", "num_stages"]
    parts = []
    for k in order:
        if k in d:
            parts.append(f'"{k}": {d[k]}')
    for k in d:
        if k not in order:
            parts.append(f'"{k}": {d[k]}')
    return "{" + ", ".join(parts) + "}"


def patch_kernel(kernel: str, entries: list[tuple[tuple, dict]]) -> bool:
    kpath = ROOT / "kernels" / kernel / "kernel.py"
    src = kpath.read_text()

    # Build the new SHAPE_CONFIGS block.
    lines = ["SHAPE_CONFIGS = {"]
    lines.append("    # Auto-tuned per-shape configs (see results/autotune_summary.csv).")
    lines.append("    # Re-generate via: python benchmarks/autotune.py --kernels " + kernel)
    for key, cfg in entries:
        lines.append(f"    {key!r}: {render_dict_literal(cfg)},")
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the existing SHAPE_CONFIGS = {...} block.
    pattern = re.compile(
        r"SHAPE_CONFIGS\s*(?::\s*[^=]+)?=\s*\{.*?^\}",
        re.DOTALL | re.MULTILINE,
    )
    if not pattern.search(src):
        print(f"  {kernel}: could not locate SHAPE_CONFIGS block; skipping", file=sys.stderr)
        return False
    new_src = pattern.sub(new_block, src, count=1)
    if new_src == src:
        print(f"  {kernel}: no change")
        return False
    kpath.write_text(new_src)
    print(f"  {kernel}: patched {kpath} with {len(entries)} configs")
    return True


def main() -> int:
    best = load_best_per_kernel()
    if not best:
        print("no best configs in CSV — nothing to apply")
        return 1
    n_changed = 0
    for kernel, entries in best.items():
        if patch_kernel(kernel, entries):
            n_changed += 1
    print(f"\npatched {n_changed} / {len(best)} kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
