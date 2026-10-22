"""#134: additive geometric correction with Warp Galerkin operators and FSAI.

Transfers are caller-supplied compact FP64 CSR matrices. Setup forms P.T H P
with upstream Warp sparse products. Apply is capturable with FP64 or FP32
working arithmetic. The underlying additive operator is fixed-linear and SPD
when its smoothers and bottom solve are SPD. No coarse eigenbasis is computed.
"""

import copy
from typing import Any

import warp as wp
import warp.sparse as ws
from warp.optim import linear

from benchmarks.experimental.coarse_correction import FixedFSAI
from warpack import kernels as K
from warpack._coalesced_fsai import (
    csr_eight_rhs,
    csr_eight_rhs_padded,
    transpose_axpby,
    transpose_rows,
)
from warpack.batched_cg import axpby
from warpack.batched_sparse import csr_block_product


def column_product(matrix, x, y):
    """Use existing Warp sparse kernels with adjacent RHS columns, including padding."""
    if matrix.row_counts is None:
        kernel = csr_eight_rhs
        args = [matrix.offsets, matrix.columns, matrix.values, x, y]
    else:
        kernel = csr_eight_rhs_padded
        args = [matrix.offsets, matrix.row_counts, matrix.columns, matrix.values, x, y]
    wp.launch_tiled(
        kernel,
        dim=((x.shape[1] + 7) // 8, matrix.nrow),
        inputs=args,
        block_dim=32,
        device=matrix.device,
    )


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
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
):
    rhs = wp.tid()
    n = lower.shape[0]
    for i in range(n):
        value = x[rhs, i]
        for j in range(i):
            value -= x.dtype(lower[i, j]) * y[rhs, j]
        y[rhs, i] = value / x.dtype(lower[i, i])
    for step in range(n):
        i = n - 1 - step
        value = y[rhs, i]
        for j in range(i + 1, n):
            value -= x.dtype(lower[j, i]) * y[rhs, j]
        y[rhs, i] = value / x.dtype(lower[i, i])
    if status[0] != 0:
        for i in range(n):
            y[rhs, i] = x.dtype(wp.nan)


def check_matrix(matrix, *, square=True):
    if (
        matrix.scalar_type != wp.float64
        or matrix.block_shape != (1, 1)
        or matrix.row_counts is not None
        or (square and matrix.shape[0] != matrix.shape[1])
    ):
        raise ValueError("compact FP64 scalar CSR required")


def check_rows(x, y, width, n, device, dtype):
    if (
        x.shape != (width, n)
        or y.shape != x.shape
        or any(a.dtype != dtype or a.device != device or not a.is_contiguous for a in (x, y))
    ):
        raise ValueError(
            "matching contiguous working-precision rows on the hierarchy device required"
        )


class DenseBottom:
    """FP64 Warp Cholesky setup, working-precision solve and changed-value replay."""

    def __init__(self, matrix, width, dtype=wp.float64):
        check_matrix(matrix)
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        if not 0 < self.n <= 64 or width < 1 or dtype not in (wp.float32, wp.float64):
            raise ValueError("bottom solve requires n<=64 and positive width")
        self.matrix = matrix
        self.dtype = dtype
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
        check_rows(x, y, self.width, self.n, self.device, self.dtype)
        wp.launch(dense_solve, self.width, [self.lower, self.status, x, y], device=self.device)

    def for_width(self, width):
        if width < 1:
            raise ValueError("positive width required")
        result = copy.copy(self)  # Bottom factor/status contain no per-RHS workspace.
        result.width = width
        return result

    def apply_columns(self, x, y):
        check_rows(x, y, self.n, self.width, self.device, self.dtype)
        # Strided views change metadata only; adjacent RHSs remain coalesced.
        wp.launch(
            dense_solve,
            self.width,
            [self.lower, self.status, x.transpose(), y.transpose()],
            device=self.device,
        )


class AdditiveLevel:
    """B = S + P B_coarse P.T; no approximate coarse eigenvectors or dense lift."""

    def __init__(self, matrix, transfer, transpose, smoother, coarse, width, *, columns=False):
        check_matrix(matrix)
        check_matrix(transfer, square=False)
        check_matrix(transpose, square=False)
        self.matrix, self.transfer, self.transpose = matrix, transfer, transpose
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        self.dtype = coarse.dtype
        self.columns = columns
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
        self.rhs = wp.empty(
            (self.n, width) if columns else (width, self.n),
            dtype=self.dtype,
            device=self.device,
        )
        self.restricted = wp.empty(
            (coarse.n, width) if columns else (width, coarse.n),
            dtype=self.dtype,
            device=self.device,
        )
        self.solved = wp.empty_like(self.restricted)
        self.correction = wp.empty_like(self.rhs)

    def apply(self, x, y):
        check_rows(x, y, self.width, self.n, self.device, self.dtype)
        if self.columns:
            wp.launch_tiled(
                transpose_rows,
                dim=((self.width + 31) // 32, (self.n + 31) // 32),
                inputs=[x, self.rhs],
                block_dim=128,
                device=self.device,
            )
            result = self.smoother.application.second
            self.apply_columns(self.rhs, result)
            wp.launch_tiled(
                transpose_axpby,
                dim=((self.n + 31) // 32, (self.width + 31) // 32),
                inputs=[result, y, y, wp.float64(1), wp.float64(0)],
                block_dim=128,
                device=self.device,
            )
            return
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

    def apply_columns(self, x, y):
        if not self.columns:
            raise ValueError("column application requires a column-layout hierarchy")
        check_rows(x, y, self.n, self.width, self.device, self.dtype)
        column_product(self.transpose, x, self.restricted)
        self.coarse.apply_columns(self.restricted, self.solved)
        column_product(self.transfer, self.solved, self.correction)
        scratch = self.smoother.application.first
        column_product(self.smoother.factor.G, x, scratch)
        column_product(self.smoother.factor.GT, scratch, y)
        wp.launch(
            axpby,
            y.size,
            [self.correction.flatten(), y.flatten(), y.flatten(), wp.float64(1), wp.float64(1)],
            device=self.device,
        )

    def for_width(self, width):
        return AdditiveLevel(
            self.matrix,
            self.transfer,
            self.transpose,
            FixedFSAI(self.smoother.factor, width, self.dtype),
            self.coarse.for_width(width),
            width,
            columns=self.columns,
        )


def build_hierarchy(
    matrix, transfers, width, *, fine_factor=None, coarse_rows=32, dtype=wp.float64, columns=False
):
    """Construct static Galerkin levels entirely with Warp numerical operations.

    A fixed hierarchy remains a valid SPD preconditioner for another SPD fine
    matrix, but is not automatically refitted. Rebuild for changed topology or
    when refreshing all coarse operators/factors is desired.
    """
    check_matrix(matrix)
    if not transfers:
        return DenseBottom(matrix, width, dtype)
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
    coarse = build_hierarchy(
        coarse_matrix, transfers[1:], width, coarse_rows=coarse_rows, dtype=dtype, columns=columns
    )
    factor = (
        fine_factor
        if fine_factor is not None
        else linear.FSAI(matrix, max_row_size=coarse_rows, max_step_size=3, factor_dtype=wp.float32)
    )
    return AdditiveLevel(
        matrix, p, pt, FixedFSAI(factor, width, dtype), coarse, width, columns=columns
    )
