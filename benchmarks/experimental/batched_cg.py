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

from warpack.generalized import _cg_preallocates_loop_control

from .block_kernels import BatchedSparseOperator
from .block_kernels import csr_block_product as product


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

    def __init__(self, matrix, width, offsets, layout="rows"):
        if layout not in ("rows", "columns"):
            raise ValueError("matrix layout must be rows or columns")
        self.coalesced = None
        if layout == "columns":
            from .coalesced_operator import CoalescedSparseApplication

            self.coalesced = CoalescedSparseApplication(matrix, width)
        self.matrix, self.width, self.n = matrix, width, matrix.shape[0]
        self.operator = BatchedSparseOperator(matrix) if matrix.scalar_type == wp.float64 else None
        self.tmp = (
            wp.empty(width * self.n, dtype=matrix.scalar_type, device=matrix.device)
            if self.coalesced is None
            else None
        )
        self.linear = linear.LinearOperator(
            (width * self.n, width * self.n),
            matrix.scalar_type,
            matrix.device,
            self.matvec,
            batch_offsets=offsets,
            max_batch_length=self.n,
        )

    def matvec(self, x, y, z, alpha, beta):
        if self.coalesced is not None:
            return self.coalesced.matvec(x, y, z, alpha, beta)
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
    def __init__(self, preconditioner, width, n, offsets, device, dtype=wp.float64, layout="rows"):
        self.preconditioner, self.width, self.n = preconditioner, width, n
        self.coalesced = None
        if layout not in ("rows", "columns"):
            raise ValueError("FSAI layout must be rows or columns")
        if layout == "columns":
            from .coalesced_fsai import CoalescedFactorApplication

            self.coalesced = CoalescedFactorApplication(preconditioner, width, n, device, dtype)
        self.tmp = (
            wp.empty((width, n), dtype=dtype, device=device) if self.coalesced is None else None
        )
        self.out = wp.empty_like(self.tmp) if self.tmp is not None else None
        self.linear = linear.LinearOperator(
            (width * n, width * n),
            dtype,
            device,
            self.matvec,
            batch_offsets=offsets,
            max_batch_length=n,
        )

    def matvec(self, x, y, z, alpha, beta):
        if self.coalesced is not None:
            return self.coalesced.matvec(x, y, z, alpha, beta)
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

    FP32 working values refresh from the FP64 source on every application,
    including graph replay. Sparse topology and captured buffer identities
    must remain fixed. Updating/rebuilding a supplied factor is the caller's
    responsibility; the working matrix itself must never silently remain old.
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
        _working_matrix=None,
        fsai_layout="rows",
        matrix_layout="rows",
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
        if fsai_layout not in ("rows", "columns"):
            raise ValueError("FSAI layout must be rows or columns")
        if fsai_layout == "columns" and (
            preconditioner is None or not hasattr(preconditioner, "G")
        ):
            raise ValueError("column layout requires upstream FSAI factors")
        if matrix_layout not in ("rows", "columns"):
            raise ValueError("matrix layout must be rows or columns")
        self.matrix_layout = matrix_layout
        self.fsai_layout = fsai_layout
        self.working_dtype = working_dtype
        self.source_matrix = matrix
        self.tol, self.atol, self.maxiter = tol, atol, maxiter
        self.single_preconditioner = preconditioner
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        self.offsets = wp.array(
            [j * self.n for j in range(width + 1)], dtype=wp.int32, device=self.device
        )
        working_matrix = _working_matrix
        if working_matrix is None:
            working_matrix = (
                matrix
                if working_dtype == wp.float64
                else ws.bsr_copy(matrix, scalar_type=working_dtype)
            )
        elif (
            working_matrix.shape != matrix.shape
            or working_matrix.block_shape != matrix.block_shape
            or working_matrix.scalar_type != working_dtype
            or working_matrix.device != matrix.device
            or working_matrix.row_counts is not None
        ):
            raise ValueError("incompatible shared working matrix")
        self.operator = RepeatedOperator(working_matrix, width, self.offsets, matrix_layout)
        self.preconditioner = (
            RepeatedPreconditioner(
                preconditioner, width, self.n, self.offsets, self.device, working_dtype, fsai_layout
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

        self.requires_static_outer_capture = not _cg_preallocates_loop_control(self.state)

    def for_width(self, width):
        """Independent CG workspaces sharing matrix topology/storage and factor."""
        return BatchedCGInverse(
            self.source_matrix,
            width,
            tol=self.tol,
            atol=self.atol,
            maxiter=self.maxiter,
            preconditioner=self.single_preconditioner,
            working_dtype=self.working_dtype,
            _working_matrix=self.operator.matrix,
            fsai_layout=self.fsai_layout,
            matrix_layout=self.matrix_layout,
        )

    def reset_stats(self):
        self.stats.zero_()

    def refresh(self):
        """Refresh the reduced-precision copy using current source values on device."""
        if self.working_dtype != self.source_matrix.scalar_type:
            source, target = self.source_matrix.values, self.operator.matrix.values
            # Copy live capacity common to both matrices. Extra allocator
            # capacity beyond the CSR/BSR offsets is not part of the matrix.
            wp.launch(cast, min(source.size, target.size), [source, target], device=self.device)

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
        self.refresh()
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


@wp.kernel
def merge_statistics(source: wp.array[wp.int32], target: wp.array[wp.int32]):
    i = wp.tid()
    if i == 1:
        wp.atomic_max(target, i, source[i])
    else:
        wp.atomic_add(target, i, source[i])


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
