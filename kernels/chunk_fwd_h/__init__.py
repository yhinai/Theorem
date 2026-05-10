"""chunk_fwd_h: gated DeltaNet inter-chunk state recurrence on AMD MI300X."""

from .kernel import custom_kernel
from .reference import ref_kernel

__all__ = ["custom_kernel", "ref_kernel"]
