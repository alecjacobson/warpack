"""Benchmark-only fresh metric check before reusing a Ritz basis (#183)."""

import math

import warp as wp

from warpack import BatchedSparseOperator, BlockLOBPCG, SparseOperator
from warpack.block import Identity


@wp.kernel
def metric_defect(
    gram: wp.array2d[wp.float64],
    bound: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
):
    i = wp.tid()
    row, column = wp.float64(0.0), wp.float64(0.0)
    for j in range(gram.shape[0]):
        diagonal = wp.float64(i == j)
        a, b = gram[i, j] - diagonal, gram[j, i] - diagonal
        if not wp.isfinite(a) or not wp.isfinite(b):
            wp.atomic_max(invalid, 0, 1)
        row += wp.abs(a)
        column += wp.abs(b)
    # Inflate the computed row/column sums against summation roundoff. This
    # bounds the computed metric defect; it is not an error bound for forming
    # the physical matrix product itself.
    inflation = wp.float64(1.0) + wp.float64(8.0 * 2.220446049250313e-16) * wp.float64(
        gram.shape[0]
    )
    wp.atomic_max(bound, 0, wp.max(row, column) * inflation)


@wp.kernel
def choose_qr(
    bound: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
    status: wp.array[wp.int32],
    fallback: wp.array[wp.int32],
    usage: wp.array[wp.int32],
    tolerance: wp.float64,
):
    reuse = invalid[0] == 0 and status[0] == 0 and wp.isfinite(bound[0]) and bound[0] <= tolerance
    fallback[0] = int(not reuse)
    if reuse:
        usage[0] += 1
    else:
        usage[1] += 1


class MetricQRReuse:
    def __init__(self, owner, storage, tolerance):
        self.owner, self.tolerance = owner, tolerance
        self.bound, self.invalid, self.fallback, self.usage = storage
        mass = owner.mass
        self.eligible = (
            owner.device.is_cuda
            and wp.is_conditional_graph_supported()
            and type(mass) in (SparseOperator, BatchedSparseOperator, Identity)
            and getattr(mass.apply, "__func__", None) is type(mass).apply
        )

    def full_qr(self):
        BlockLOBPCG._orthogonalize_x_full(self.owner)

    def apply(self):
        owner = self.owner
        if not self.eligible or not wp.get_stream(owner.device).is_capturing:
            self.full_qr()
            return
        owner.mass.apply(owner.sx, owner.bx)
        owner.gram(owner.sx, owner.bx, owner.gx, symmetric=True)
        self.bound.zero_()
        self.invalid.zero_()
        wp.launch(metric_defect, owner.b, [owner.gx, self.bound, self.invalid], device=owner.device)
        wp.launch(
            choose_qr,
            1,
            [self.bound, self.invalid, owner.status, self.fallback, self.usage, self.tolerance],
            device=owner.device,
        )
        wp.capture_if(self.fallback, self.full_qr)


class MetricQRReuseBlock(BlockLOBPCG):
    _ritz_qr_enabled = False
    _metric_reuse_tolerance = 1e-13
    _metric_reuse_minimum = 1
    _metric_reuse_maximum = 2147483647

    def __init__(self, *args, **kwargs):
        tolerance = float(self._metric_reuse_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0 or tolerance > 1e-12:
            raise ValueError("require an explicit finite metric-defect tolerance in (0, 1e-12]")
        super().__init__(*args, **kwargs)
        self.metric_reuse_usage = None
        self._ritz_qr_reuse = None
        if not self._metric_reuse_minimum <= self.b <= self._metric_reuse_maximum:
            return
        storage = [wp.zeros(1, dtype=wp.float64, device=self.device)] + [
            wp.zeros(n, dtype=wp.int32, device=self.device) for n in (1, 1, 2)
        ]
        self.metric_reuse_usage = storage[-1]
        # Branches execute mutually exclusively and may share these 24 bytes.
        for branch in [self] + ([] if self._search is None else self._search.buckets):
            branch._ritz_qr_reuse = MetricQRReuse(branch, storage, tolerance)

    def initialize(self, seed=42):
        if self.metric_reuse_usage is not None:
            self.metric_reuse_usage.zero_()
        super().initialize(seed)

    def orthogonalize_x(self):
        helper = getattr(self, "_ritz_qr_reuse", None)
        if helper is None:
            super().orthogonalize_x()
        else:
            helper.apply()
