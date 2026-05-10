"""Gated DeltaNet WY-transform recompute of (w, u) on AMD MI300X.

Exports:
    custom_kernel(data) -> (w, u)   — Triton (AMD backend) implementation
    ref_kernel(data)    -> (w, u)   — pure PyTorch reference
"""

from .kernel import custom_kernel, SHAPE_CONFIGS
from .reference import ref_kernel, generate_data, Data, BT

__all__ = [
    "custom_kernel",
    "ref_kernel",
    "generate_data",
    "Data",
    "BT",
    "SHAPE_CONFIGS",
]
