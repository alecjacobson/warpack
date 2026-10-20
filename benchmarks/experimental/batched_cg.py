"""Reference adapters for CuPy comparisons; numerical CG lives in warpack."""

import warp as wp

from warpack.batched_cg import (
    RepeatedOperator as RepeatedOperator,
)
from warpack.batched_cg import (
    RepeatedPreconditioner as RepeatedPreconditioner,
)
from warpack.batched_cg import (
    merge_statistics,
)


class CapturedPaddedCG:
    """Benchmark adapter for CuPy's shrinking active block, sharing Warp CG state.

    Zero-pad to the configured batch width and reuse one captured inner solve.
    The caller must serialize this adapter with any other use of the inverse.
    Stats include padded zero RHSs and masked iteration slots.
    """

    def __init__(self, inverse):
        self.inverse = inverse
        self.rhs = wp.zeros((inverse.width, inverse.n), dtype=wp.float64, device=inverse.device)
        self.solution = wp.empty_like(self.rhs)
        inverse.apply(self.rhs, self.solution)
        wp.synchronize_device(inverse.device)
        with wp.ScopedCapture(device=inverse.device) as capture:
            inverse.apply(self.rhs, self.solution)
        self.graph = capture.graph

    def apply(self, x, y):
        if x.shape != y.shape or len(x.shape) != 2 or x.shape[1] != self.inverse.n:
            raise ValueError("expected matching (active_width,n) arrays")
        if not 1 <= x.shape[0] <= self.inverse.width:
            raise ValueError("active block exceeds captured batch width")
        if x.dtype != wp.float64 or y.dtype != wp.float64:
            raise ValueError("expected FP64 arrays")
        if x.device != self.inverse.device or y.device != self.inverse.device:
            raise ValueError("expected arrays on the inverse device")
        if x.shape[0] < self.inverse.width:
            self.rhs.zero_()
        wp.copy(self.rhs[: x.shape[0]], x)
        wp.capture_launch(self.graph)
        wp.copy(y, self.solution[: x.shape[0]])


class CapturedActiveCG:
    """Preallocate every active block width; never compute padded zero RHSs.

    The shifted working matrix and FSAI factors are shared. CG and graph buffers
    are independent per width, so setup memory grows with sum(1..width).
    This benchmark adapter is serialized with other uses of the source inverse.
    """

    def __init__(self, inverse, max_width=None):
        width = inverse.width if max_width is None else max_width
        if not 1 <= width <= inverse.width:
            raise ValueError("require 1 <= max_width <= inverse.width")
        self.adapters = {
            w: CapturedPaddedCG(inverse if w == inverse.width else inverse.for_width(w))
            for w in range(1, width + 1)
        }
        self.stats = wp.zeros(4, dtype=wp.int32, device=inverse.device)
        self.device = inverse.device

    def reset_stats(self):
        for adapter in self.adapters.values():
            adapter.inverse.reset_stats()

    def collect_stats(self):
        self.stats.zero_()
        for adapter in self.adapters.values():
            wp.launch(merge_statistics, 4, [adapter.inverse.stats, self.stats], device=self.device)
        return self.stats

    def apply(self, x, y):
        width = x.shape[0]
        if width not in self.adapters:
            raise ValueError("active block width was not preallocated")
        self.adapters[width].apply(x, y)
