"""Implicit centered boundary products for the conformal benchmark.

The factor/preconditioner can remain unchanged; only exact matrix products
are represented as diagonal minus rank-two updates. No host work on replay.
"""

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear

from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator


@wp.kernel
def boundary_sums(
    x: wp.array2d[wp.float64], indices: wp.array2d[wp.int32], sums: wp.array2d[wp.float64]
):
    row, component, chunk, lane = wp.tid()
    index = indices[component, chunk * 256 + lane]
    value = wp.float64(0.0)
    if index >= 0:
        value = x[row, index]
    total = wp.tile_sum(wp.tile(value))
    if lane == 0:
        wp.atomic_add(sums, row, component, wp.tile_extract(total, 0))


@wp.kernel
def centered_product(
    x: wp.array2d[wp.float64],
    sums: wp.array2d[wp.float64],
    indicator: wp.array[wp.int32],
    nb: wp.float64,
    y: wp.array2d[wp.float64],
):
    row, i = wp.tid()
    value = wp.float64(0.0)
    if indicator[i] != 0:
        value = x[row, i] - sums[row, i // (x.shape[1] // 2)] / nb
    y[row, i] = value


@wp.kernel
def shifted_product(
    base: wp.array2d[wp.float64],
    sums: wp.array2d[wp.float64],
    indicator: wp.array[wp.int32],
    coefficient: wp.float64,
    y: wp.array2d[wp.float64],
    z: wp.array2d[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
):
    row, i = wp.tid()
    value = base[row, i]
    if indicator[i] != 0:
        value += coefficient * sums[row, i // (base.shape[1] // 2)]
    value *= alpha
    if beta != wp.float64(0.0):
        value += beta * y[row, i]
    z[row, i] = value


class CenteredBoundary:
    def __init__(self, case, capacity):
        self.n, self.capacity = case.n, capacity
        nb, vertices = len(case.boundary), len(case.xyz)
        reverse = np.full(2 * vertices, -1, dtype=np.int32)
        reverse[case.free] = np.arange(case.n)
        ids = np.full((2, 256 * ((nb + 255) // 256)), -1, dtype=np.int32)
        ids[:, :nb] = reverse[np.array([case.boundary, case.boundary + vertices])]
        full_indicator = np.zeros(2 * vertices, dtype=np.int32)
        full_indicator[np.r_[case.boundary, case.boundary + vertices]] = 1
        self.indices = wp.array(ids, dtype=wp.int32)
        self.indicator = wp.array(full_indicator[case.free], dtype=wp.int32)
        self.nb, self.device = float(nb), self.indices.device
        self.sums = wp.zeros((capacity, 2), dtype=wp.float64, device=self.device)

    def reduce(self, x):
        if x.ndim != 2 or x.shape[1] != self.n or x.shape[0] > self.capacity:
            raise ValueError("boundary operator input exceeds its fixed shape/capacity")
        self.sums.zero_()
        wp.launch_tiled(
            boundary_sums,
            dim=(x.shape[0], 2, self.indices.shape[1] // 256),
            inputs=[x, self.indices, self.sums],
            block_dim=256,
            device=self.device,
        )

    def apply(self, x, y):
        self.reduce(x)
        wp.launch(
            centered_product,
            x.shape,
            [x, self.sums, self.indicator, self.nb, y],
            device=self.device,
        )


class ShiftedConformal:
    """A=K-sigma M, with sparse K-sigma D and a rank-two update.

    The shared sums and scratch are private per operator and ordered on one
    stream. LinearOperator matvec supports x/y/z aliases and alpha/beta.
    """

    def __init__(self, case, capacity):
        self.boundary = CenteredBoundary(case, capacity)
        k, m = case.reduced
        base = k - case.shift * sp.diags((m.diagonal() > 0).astype(float))
        self.base = SparseOperator(matrix(base))
        self.n, self.device = case.n, self.base.device
        self.scratch = wp.empty((capacity, self.n), dtype=wp.float64, device=self.device)
        self.coefficient = case.shift / self.boundary.nb
        self.linear = linear.LinearOperator((self.n, self.n), wp.float64, self.device, self.matvec)

    def apply(self, x, y):
        self.boundary.reduce(x)
        scratch = self.scratch[: x.shape[0]]
        self.base.apply(x, scratch)
        wp.launch(
            shifted_product,
            x.shape,
            [
                scratch,
                self.boundary.sums,
                self.boundary.indicator,
                self.coefficient,
                y,
                y,
                1.0,
                0.0,
            ],
            device=self.device,
        )

    def matvec(self, x, y, z, alpha, beta):
        xx, yy, zz = (v.reshape((1, self.n)) for v in (x, y, z))
        self.boundary.reduce(xx)
        scratch = self.scratch[:1]
        self.base.apply(xx, scratch)
        wp.launch(
            shifted_product,
            xx.shape,
            [
                scratch,
                self.boundary.sums,
                self.boundary.indicator,
                self.coefficient,
                yy,
                zz,
                wp.float64(alpha),
                wp.float64(beta),
            ],
            device=self.device,
        )
