"""#134: additive geometric correction with Warp Galerkin operators and FSAI.

Transfers are caller-supplied compact FP64 CSR matrices. Setup forms P.T H P
with upstream Warp sparse products. Apply is capturable, fixed-linear and SPD
when its smoothers and bottom solve are SPD. No coarse eigenbasis is computed.
"""

import warp as wp
import warp.sparse as ws
from warp.optim import linear

from benchmarks.experimental.coarse_correction import FixedFSAI
from warpack import kernels as K
from warpack.batched_cg import axpby
from warpack.batched_sparse import csr_block_product


@wp.kernel
def to_dense(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    dense: wp.array2d[wp.float64],
):
    row = wp.tid()
    for e in range(offsets[row], offsets[row + 1]):
        dense[row, columns[e]] = values[e]


@wp.kernel
def dense_solve(
    lower: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    rhs = wp.tid()
    n = lower.shape[0]
    for i in range(n):
        value = x[rhs, i]
        for j in range(i):
            value -= lower[i, j] * y[rhs, j]
        y[rhs, i] = value / lower[i, i]
    for step in range(n):
        i = n - 1 - step
        value = y[rhs, i]
        for j in range(i + 1, n):
            value -= lower[j, i] * y[rhs, j]
        y[rhs, i] = value / lower[i, i]
    if status[0] != 0:
        for i in range(n):
            y[rhs, i] = wp.float64(wp.nan)


def check_matrix(matrix, *, square=True):
    if (
        matrix.scalar_type != wp.float64
        or matrix.block_shape != (1, 1)
        or matrix.row_counts is not None
        or (square and matrix.shape[0] != matrix.shape[1])
    ):
        raise ValueError("compact FP64 scalar CSR required")


def check_rows(x, y, width, n, device):
    if (
        x.shape != (width, n)
        or y.shape != x.shape
        or any(a.dtype != wp.float64 or a.device != device or not a.is_contiguous for a in (x, y))
    ):
        raise ValueError("matching contiguous FP64 rows on the hierarchy device required")


class DenseBottom:
    """Small preallocated Warp Cholesky inverse, including changed-value replay."""

    def __init__(self, matrix, width):
        check_matrix(matrix)
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        if not 0 < self.n <= 64 or width < 1:
            raise ValueError("bottom solve requires n<=64 and positive width")
        self.matrix = matrix
        self.dense = wp.zeros((self.n, self.n), dtype=wp.float64, device=self.device)
        self.lower = wp.zeros_like(self.dense)
        self.status = wp.ones(1, dtype=wp.int32, device=self.device)
        self.update()

    def update(self):
        self.dense.zero_()
        self.status.zero_()
        wp.launch(
            to_dense,
            self.n,
            [self.matrix.offsets, self.matrix.columns, self.matrix.values, self.dense],
            device=self.device,
        )
        wp.launch(K.cholesky, 1, [self.dense, self.lower, self.status], device=self.device)

    def apply(self, x, y):
        check_rows(x, y, self.width, self.n, self.device)
        wp.launch(dense_solve, self.width, [self.lower, self.status, x, y], device=self.device)


class AdditiveLevel:
    """B = S + P B_coarse P.T; no approximate coarse eigenvectors or dense lift."""

    def __init__(self, matrix, transfer, transpose, smoother, coarse, width):
        check_matrix(matrix)
        check_matrix(transfer, square=False)
        check_matrix(transpose, square=False)
        self.matrix, self.transfer, self.transpose = matrix, transfer, transpose
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        if (
            transfer.shape != (self.n, coarse.n)
            or transpose.shape != (coarse.n, self.n)
            or transfer.device != self.device
            or transpose.device != self.device
            or coarse.device != self.device
            or coarse.width != width
            or smoother.n != self.n
            or smoother.device != self.device
        ):
            raise ValueError("incompatible transfer, smoother or coarse level")
        self.smoother, self.coarse = smoother, coarse
        self.status = coarse.status
        self.rhs = wp.empty((width, self.n), dtype=wp.float64, device=self.device)
        self.restricted = wp.empty((width, coarse.n), dtype=wp.float64, device=self.device)
        self.solved = wp.empty_like(self.restricted)
        self.correction = wp.empty_like(self.rhs)

    def apply(self, x, y):
        check_rows(x, y, self.width, self.n, self.device)
        wp.copy(self.rhs, x)  # Preserve the original RHS for exact in-place application.
        csr_block_product(self.transpose, self.rhs, self.restricted)
        self.coarse.apply(self.restricted, self.solved)
        csr_block_product(self.transfer, self.solved, self.correction)
        self.smoother.apply(self.rhs, y)
        wp.launch(
            axpby,
            y.size,
            [self.correction.flatten(), y.flatten(), y.flatten(), wp.float64(1), wp.float64(1)],
            device=self.device,
        )


def build_hierarchy(matrix, transfers, width, *, fine_factor=None, coarse_rows=32):
    """Construct static Galerkin levels entirely with Warp numerical operations.

    A fixed hierarchy remains a valid SPD preconditioner for another SPD fine
    matrix, but is not automatically refitted. Rebuild for changed topology or
    when refreshing all coarse operators/factors is desired.
    """
    check_matrix(matrix)
    if not transfers:
        return DenseBottom(matrix, width)
    p = transfers[0]
    check_matrix(p, square=False)
    if (
        p.device != matrix.device
        or p.shape[0] != matrix.shape[0]
        or not 0 < p.shape[1] < p.shape[0]
    ):
        raise ValueError("transfer must strictly coarsen on the same device")
    pt = ws.bsr_transposed(p)
    hp = ws.bsr_mm(matrix, p)
    coarse_matrix = ws.bsr_mm(pt, hp)
    # Sparse accumulation orders can differ; explicitly retain symmetry.
    ws.bsr_axpy(ws.bsr_transposed(coarse_matrix), coarse_matrix, alpha=0.5, beta=0.5)
    coarse = build_hierarchy(coarse_matrix, transfers[1:], width, coarse_rows=coarse_rows)
    factor = (
        fine_factor
        if fine_factor is not None
        else linear.FSAI(matrix, max_row_size=coarse_rows, max_step_size=3, factor_dtype=wp.float32)
    )
    return AdditiveLevel(matrix, p, pt, FixedFSAI(factor, width), coarse, width)
