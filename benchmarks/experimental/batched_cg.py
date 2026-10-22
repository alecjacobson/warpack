"""Reference adapters for CuPy comparisons; numerical CG lives in warpack."""

import warp as wp

from warpack import BatchedCGPreconditioner
from warpack.batched_cg import (
    RepeatedOperator as RepeatedOperator,
)
from warpack.batched_cg import (
    RepeatedPreconditioner as RepeatedPreconditioner,
)
from warpack.batched_cg import (
    merge_statistics,
)

BatchedCGInverse = BatchedCGPreconditioner


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


class CapturedPartitionedCG(CapturedActiveCG):
    """Exact RHS coverage using power-of-two chunks and linear workspace storage.

    A width-104 call runs 64, 32 and 8 independent RHSs. No padded zero RHSs
    are computed. Each chunk stops independently using unchanged upstream CG;
    this can reduce scheduled masked work compared with one global active block.
    At most floor(log2(max_width))+1 graphs are launched per call. All numerical
    work remains Warp; host chunk selection is for the CuPy benchmark adapter.
    """

    def __init__(self, inverse, max_width=None):
        self.width = inverse.width if max_width is None else max_width
        if not isinstance(self.width, int) or not 1 <= self.width <= inverse.width:
            raise ValueError("require integer 1 <= max_width <= inverse.width")
        self.n, self.device = inverse.n, inverse.device
        self.adapters = {}
        for exponent in range(self.width.bit_length()):
            width = 1 << exponent
            item = CapturedPaddedCG(inverse if width == inverse.width else inverse.for_width(width))
            # Finish lazy graph uploads during setup, not while a preceding
            # chunk is pending on the #61 driver stack.
            wp.capture_launch(item.graph)
            wp.synchronize_device(self.device)
            self.adapters[width] = item
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        # Preserve the entire input before any output, including shifted aliases.
        self.rhs = wp.empty((self.width, self.n), dtype=wp.float64, device=self.device)

    def apply(self, x, y):
        if (
            x.shape != y.shape
            or x.ndim != 2
            or x.shape[1] != self.n
            or not 1 <= x.shape[0] <= self.width
            or x.dtype != wp.float64
            or y.dtype != wp.float64
            or x.device != self.device
            or y.device != self.device
        ):
            raise ValueError("expected matching FP64 (active_width,n) arrays on the inverse device")
        wp.copy(self.rhs[: x.shape[0]], x)
        offset, remaining = 0, x.shape[0]
        for width, adapter in reversed(self.adapters.items()):
            if remaining >= width:
                adapter.apply(self.rhs[offset : offset + width], y[offset : offset + width])
                offset += width
                remaining -= width
