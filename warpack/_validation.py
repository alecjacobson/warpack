"""Metadata-only validation for public operator inputs; no device reads."""

import warp as wp


def fp64_rows(x, y, n, device, *, width=None, contiguous=False):
    """Check row storage before launching a kernel or passing native pointers."""
    if x.ndim != 2 or x.shape[1] != n or x.shape[0] < 1 or y.shape != x.shape:
        raise ValueError("expected matching nonempty (width, n) rows")
    if width is not None and x.shape[0] != width:
        raise ValueError("RHS width must match the width used at setup")
    if x.dtype != wp.float64 or y.dtype != wp.float64:
        raise TypeError("operator rows must have dtype FP64")
    if x.device != device or y.device != device:
        raise ValueError("operator rows must be on the operator device")
    if contiguous and (not x.is_contiguous or not y.is_contiguous):
        raise ValueError("cuDSS rows must be contiguous")
