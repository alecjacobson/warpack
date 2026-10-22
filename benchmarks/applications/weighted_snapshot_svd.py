"""Checkout-only lumped-mass ROM composition using public augmented PartialSVD.

Caller-provided physical response rows G are whitened as sqrt(M) G on device.
The three captured stages prepare snapshots, compute singular triplets, then
recover physical modes and covariance values. No Gram matrix is formed.
"""

import numpy as np
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import PartialSVD


@wp.kernel
def weight_snapshots(
    rows: wp.array2d[wp.float64],
    mass: wp.array[wp.float64],
    values: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
):
    i, j = wp.tid()
    g, m = rows[j, i], mass[i]
    value = wp.float64(0)
    if wp.isfinite(g) and wp.isfinite(m) and m > wp.float64(0):
        value = wp.sqrt(m) * g
        if not wp.isfinite(value):
            value = wp.float64(0)
            wp.atomic_max(invalid, 0, 1)
    else:
        wp.atomic_max(invalid, 0, 1)
    values[i * rows.shape[0] + j] = value


@wp.kernel
def physical_modes(
    left: wp.array2d[wp.float64],
    singular: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
    cutoff: wp.float64,
    modes: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    positive: wp.array[wp.int32],
):
    j, i = wp.tid()
    s = singular[j]
    valid = invalid[0] == 0 and wp.isfinite(s) and s > cutoff * singular[0] and s > wp.float64(0)
    modes[j, i] = wp.float64(0)
    if valid:
        modes[j, i] = left[j, i] / wp.sqrt(mass[i])
    if i == 0:
        values[j] = wp.float64(0)
        if valid:
            values[j] = s * s
            wp.atomic_add(positive, 0, 1)


@wp.kernel
def output_status(
    invalid: wp.array[wp.int32],
    native: wp.array[wp.int32],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    positive: wp.array[wp.int32],
    k: int,
    status: wp.array[wp.int32],
):
    status[0] = native[0]
    if invalid[0] != 0:
        status[0] = 1
    elif positive[0] != k or left[0] != k or right[0] != k:
        status[0] = wp.max(status[0], 2)


class LumpedSnapshotSVD:
    """Positive covariance modes; fixed addresses, mutable G and positive mass.

    This is a lumped-mass application wrapper, not a general weighted-SVD API.
    Dense snapshot storage is O(n*r); it never constructs n-by-n covariance.
    Source H solves, if needed, precede this composition and must be validated
    independently. Its singular-value rank cutoff is distinct from Gram rank.
    """

    def __init__(self, responses, mass, k, *, tol=1e-12, rank_tol=1e-12, ncv=None):
        if (
            responses.ndim != 2
            or responses.dtype != wp.float64
            or mass.dtype != wp.float64
            or mass.shape != (responses.shape[1],)
            or responses.device != mass.device
            or not responses.is_contiguous
            or not mass.is_contiguous
            or not 0 < rank_tol < 1
        ):
            raise ValueError(
                "contiguous FP64 response rows and diagonal mass on one device required"
            )
        self.r, self.n = responses.shape
        if not 1 <= k <= min(self.r, self.n):
            raise ValueError("positive mode count cannot exceed snapshot dimensions")
        self.responses, self.mass, self.k = responses, mass, k
        self.device, self.rank_tol = responses.device, rank_tol
        # Host topology allocation only. Numerical values are always prepared in Warp.
        with wp.ScopedDevice(self.device):
            self.weighted = matrix(np.ones((self.n, self.r)))
        self.invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.status = wp.zeros_like(self.invalid)
        self.positive = wp.zeros_like(self.invalid)
        self.values = wp.empty(k, dtype=wp.float64, device=self.device)
        self.modes = wp.empty((k, self.n), dtype=wp.float64, device=self.device)
        self.solver = PartialSVD(self.weighted, k, ncv=ncv, tol=tol)
        self.graphs = None

    def prepare(self):
        self.invalid.zero_()
        wp.launch(
            weight_snapshots,
            (self.n, self.r),
            [self.responses, self.mass, self.weighted.values, self.invalid],
            device=self.device,
        )

    def recover(self):
        self.positive.zero_()
        wp.launch(
            physical_modes,
            self.modes.shape,
            [
                self.solver.u,
                self.solver.singular_values,
                self.mass,
                self.invalid,
                self.rank_tol,
                self.modes,
                self.values,
                self.positive,
            ],
            device=self.device,
        )
        wp.launch(
            output_status,
            1,
            [
                self.invalid,
                self.solver.status,
                self.solver.count_left,
                self.solver.count_right,
                self.positive,
                self.k,
                self.status,
            ],
            device=self.device,
        )

    def capture(self, iterations=100):
        self.prepare()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self.prepare()
        preparation = capture.graph
        svd = self.solver.capture(iterations)
        self.recover()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self.recover()
        self.graphs = (preparation, svd, capture.graph)
        # First launch instantiates/uploads each graph. Complete these separately
        # at setup: lazy upload while a preceding cuDSS response is pending can
        # corrupt the first weighted snapshot on the #61/#148 driver stack.
        # Subsequent solve() calls enqueue all stages without host synchronization.
        for graph in self.graphs:
            wp.capture_launch(graph)
            wp.synchronize_device(self.device)
        return self.graphs

    def solve(self):
        if self.graphs is None:
            raise ValueError("capture before replay")
        for graph in self.graphs:
            wp.capture_launch(graph)
