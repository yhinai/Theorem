"""chunk_fwd_o: gated DeltaNet chunkwise output (forward, o) for AMD MI300X."""

from .kernel import custom_kernel
from .reference import ref_kernel

__all__ = ["custom_kernel", "ref_kernel"]
