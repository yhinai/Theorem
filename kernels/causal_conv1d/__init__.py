"""Depthwise causal 1D convolution kernel for AMD Instinct MI300X."""
from .kernel import custom_kernel
from .reference import ref_kernel

__all__ = ["custom_kernel", "ref_kernel"]
