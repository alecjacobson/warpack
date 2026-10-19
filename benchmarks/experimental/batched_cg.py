"""Experimental multiple-RHS adapter for upstream Warp's batched CG.

Each RHS has independent CG coefficients and residual tests via batch_offsets.
This is not block CG and does not implement a separate iterative algorithm.
FSAI factors come from upstream Warp; only their block application is batched.
"""

import functools
from typing import Any

import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from .block_kernels import BatchedSparseOperator


@wp.kernel(module="unique")
def csr_product(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array(dtype=Any),
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = x.dtype(0.0)
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row + 1], 8):
            value += x.dtype(values[e]) * x[rhs, columns[e]]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row] = wp.tile_extract(totals, lane // 8)


@wp.kernel
def axpby(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    alpha: wp.float64,
    beta: wp.float64,
):
    i = wp.tid()
    value = x.dtype(alpha) * x[i]
    if beta != wp.float64(0.0):
        value += x.dtype(beta) * y[i]
    z[i] = value


@wp.kernel
def accumulate_statistics(
    iterations: wp.array[wp.int32],
    residual_sq: wp.array(dtype=Any),
    tolerance_sq: wp.array(dtype=Any),
    stats: wp.array[wp.int32],
):
    rhs = wp.tid()
    # This counts scheduled vector iterations, including masked converged RHSs.
    wp.atomic_add(stats, 0, iterations[0])
    wp.atomic_max(stats, 1, iterations[0])
    if residual_sq[rhs] > tolerance_sq[rhs] or not wp.isfinite(residual_sq[rhs]):
        wp.atomic_add(stats, 2, 1)
    wp.atomic_add(stats, 3, 1)


def product(matrix, x, y):
    wp.launch_tiled(
        csr_product,
        dim=((x.shape[0] + 3) // 4, matrix.nrow),
        inputs=[matrix.offsets, matrix.columns, matrix.values, x, y],
        block_dim=32,
        device=matrix.device,
    )


@wp.kernel
def cast(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    i = wp.tid()
    y[i] = y.dtype(x[i])


@functools.lru_cache(None)
def bsr_product(dtype):
    vector = wp.types.vector(length=3, dtype=dtype)
    matrix = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel(module="unique")
    def compute(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array(dtype=matrix),
        x: wp.array2d(dtype=dtype),
        y: wp.array2d(dtype=dtype),
    ):
        group, row, lane = wp.tid()
        rhs, slot = group * 4 + lane // 8, lane % 8
        value = vector(dtype(0.0))
        if rhs < x.shape[0]:
            for e in range(offsets[row] + slot, offsets[row + 1], 8):
                col = columns[e] * 3
                value += values[e] * vector(x[rhs, col], x[rhs, col + 1], x[rhs, col + 2])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=1)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=1)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=1)
        if rhs < x.shape[0] and slot == 0:
            y[rhs, row * 3] = wp.tile_extract(a, lane // 8)
            y[rhs, row * 3 + 1] = wp.tile_extract(b, lane // 8)
            y[rhs, row * 3 + 2] = wp.tile_extract(c, lane // 8)

    return compute


class RepeatedOperator:
    """Same sparse operator on independent contiguous vector blocks."""

    def __init__(self, matrix, width, offsets):
        self.matrix, self.width, self.n = matrix, width, matrix.shape[0]
        self.operator = BatchedSparseOperator(matrix) if matrix.scalar_type == wp.float64 else None
        self.tmp = wp.empty(width * self.n, dtype=matrix.scalar_type, device=matrix.device)
        self.linear = linear.LinearOperator(
            (self.tmp.size, self.tmp.size),
            matrix.scalar_type,
            matrix.device,
            self.matvec,
            batch_offsets=offsets,
            max_batch_length=self.n,
        )

    def matvec(self, x, y, z, alpha, beta):
        output = z if alpha == 1 and beta == 0 and z.ptr != x.ptr else self.tmp
        rows, out = x.reshape((self.width, self.n)), output.reshape((self.width, self.n))
        if self.matrix.block_shape == (1, 1):
            product(self.matrix, rows, out)
        elif self.operator is not None:
            self.operator.apply(rows, out)
        else:
            wp.launch_tiled(
                bsr_product(self.matrix.scalar_type),
                dim=((self.width + 3) // 4, self.matrix.nrow),
                inputs=[self.matrix.offsets, self.matrix.columns, self.matrix.values, rows, out],
                block_dim=32,
                device=z.device,
            )
        if output.ptr != z.ptr:
            wp.launch(
                axpby, z.size, [output, y, z, wp.float64(alpha), wp.float64(beta)], device=z.device
            )


class RepeatedPreconditioner:
    def __init__(self, preconditioner, width, n, offsets, device, dtype=wp.float64):
        self.preconditioner, self.width, self.n = preconditioner, width, n
        self.tmp = wp.empty((width, n), dtype=dtype, device=device)
        self.out = wp.empty_like(self.tmp)
        self.linear = linear.LinearOperator(
            (width * n, width * n),
            dtype,
            device,
            self.matvec,
            batch_offsets=offsets,
            max_batch_length=n,
        )

    def matvec(self, x, y, z, alpha, beta):
        pre = self.preconditioner
        rows = x.reshape((self.width, self.n))
        if hasattr(pre, "G") and hasattr(pre, "GT"):
            # Factorization, sparsity and precision policy are upstream FSAI's.
            product(pre.G, rows, self.tmp)
            product(pre.GT, self.tmp, self.out)
            wp.launch(
                axpby,
                z.size,
                [self.out.flatten(), y, z, wp.float64(alpha), wp.float64(beta)],
                device=z.device,
            )
        else:
            for rhs in range(self.width):
                sl = slice(rhs * self.n, (rhs + 1) * self.n)
                pre.matvec(x[sl], y[sl], z[sl], alpha, beta)


class BatchedCGInverse:
    """Preallocated upstream batched CG with independent per-RHS tolerances.

    stats: scheduled vector iterations, maximum batch iterations, unconverged
    RHSs, RHS calls. Scheduled iterations include masked already-converged RHSs;
    they are not the sum of independent first-convergence iteration counts.
    """

    requires_static_outer_capture = True

    def __init__(
        self,
        matrix,
        width,
        tol=1e-12,
        maxiter=1000,
        preconditioner=None,
        atol=0.0,
        working_dtype=wp.float64,
    ):
        if not hasattr(linear.LinearOperator, "batch_offsets"):
            raise RuntimeError("Requires upstream Warp batched LinearOperator/CG support")
        if width < 1 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("require positive width and a square matrix")
        if matrix.scalar_type != wp.float64 or matrix.block_shape not in ((1, 1), (3, 3)):
            raise ValueError("require scalar or 3x3 FP64 sparse matrix")
        if matrix.row_counts is not None:
            raise ValueError("Call warp.sparse.bsr_compress before constructing an inverse")
        if working_dtype not in (wp.float32, wp.float64):
            raise ValueError("working_dtype must be float32 or float64")
        if (
            preconditioner is not None
            and not hasattr(preconditioner, "G")
            and preconditioner.scalar_type != working_dtype
        ):
            raise ValueError("preconditioner must match working precision, or be upstream FSAI")
        self.working_dtype = working_dtype
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        self.offsets = wp.array(
            [j * self.n for j in range(width + 1)], dtype=wp.int32, device=self.device
        )
        working_matrix = (
            matrix
            if working_dtype == wp.float64
            else ws.bsr_copy(matrix, scalar_type=working_dtype)
        )
        self.operator = RepeatedOperator(working_matrix, width, self.offsets)
        self.preconditioner = (
            RepeatedPreconditioner(
                preconditioner, width, self.n, self.offsets, self.device, working_dtype
            )
            if preconditioner is not None
            else None
        )
        self.b = wp.zeros(width * self.n, dtype=working_dtype, device=self.device)
        self.x = wp.zeros_like(self.b)
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.state = linear.cg(
            self.operator.linear,
            self.b,
            self.x,
            M=None if self.preconditioner is None else self.preconditioner.linear,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            check_every=0,
            run=False,
        )

    def reset_stats(self):
        self.stats.zero_()

    def apply(self, x, y):
        if x.shape != (self.width, self.n) or y.shape != x.shape:
            raise ValueError("RHS arrays must match the configured (width,n) shape")
        if (
            x.dtype != wp.float64
            or y.dtype != wp.float64
            or x.device != self.device
            or y.device != self.device
        ):
            raise ValueError("RHS arrays must be FP64 on the matrix device")
        if self.working_dtype == wp.float64:
            wp.copy(self.b, x.flatten())
        else:
            wp.launch(cast, self.b.size, [x.flatten(), self.b], device=self.device)
        self.x.zero_()
        iterations, residual_sq, tolerance_sq = self.state()
        if self.working_dtype == wp.float64:
            wp.copy(y.flatten(), self.x)
        else:
            wp.launch(cast, self.x.size, [self.x, y.flatten()], device=self.device)
        wp.launch(
            accumulate_statistics,
            self.width,
            [iterations, residual_sq, tolerance_sq, self.stats],
            device=self.device,
        )
