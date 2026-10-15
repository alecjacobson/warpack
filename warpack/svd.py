"""Largest singular triplets via a symmetric augmented operator, without A^T A."""

import warp as wp
import warp.sparse as ws

from . import kernels as K
from .solver import KrylovSchur


@wp.kernel
def augmented_mv(
    ar: wp.array[wp.int32],
    ac: wp.array[wp.int32],
    av: wp.array[wp.float64],
    br: wp.array[wp.int32],
    bc: wp.array[wp.int32],
    bv: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    nrow: int,
):
    j, i = wp.tid()
    val = wp.float64(0.0)
    if i < nrow:
        for e in range(ar[i], ar[i + 1]):
            val += av[e] * x[j, nrow + ac[e]]
    else:
        r = i - nrow
        for e in range(br[r], br[r + 1]):
            val += bv[e] * x[j, bc[e]]
    y[j, i] = val


@wp.kernel
def augmented_mv_cooperative(
    ar: wp.array[wp.int32],
    ac: wp.array[wp.int32],
    av: wp.array[wp.float64],
    br: wp.array[wp.int32],
    bc: wp.array[wp.int32],
    bv: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    nrow: int,
):
    j, i, lane = wp.tid()
    val = wp.float64(0.0)
    if i < nrow:
        for e in range(ar[i] + lane, ar[i + 1], 32):
            val += av[e] * x[j, nrow + ac[e]]
    else:
        r = i - nrow
        for e in range(br[r] + lane, br[r + 1], 32):
            val += bv[e] * x[j, bc[e]]
    value = wp.tile_extract(wp.tile_sum(wp.tile(val)), 0)
    if lane == 0:
        y[j, i] = value


@wp.kernel
def split_vectors(
    q: wp.array2d[wp.float64],
    u: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    vals: wp.array[wp.float64],
    out: wp.array[wp.float64],
    nrow: int,
):
    j, i = wp.tid()
    if i < nrow:
        u[j, i] = q[j, i]
    else:
        v[j, i - nrow] = q[j, i]
    if i == 0:
        out[j] = wp.max(wp.float64(0.0), vals[j])


@wp.kernel
def rect_mv(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    j, i = wp.tid()
    val = wp.float64(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        val += values[e] * x[j, cols[e]]
    y[j, i] = val


@wp.kernel
def rect_mv_cooperative(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    j, row, lane = wp.tid()
    value = wp.float64(0.0)
    for e in range(offsets[row] + lane, offsets[row + 1], 32):
        value += values[e] * x[j, cols[e]]
    total = wp.tile_extract(wp.tile_sum(wp.tile(value)), 0)
    if lane == 0:
        y[j, row] = total


@wp.kernel
def rect_mv_partials(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    parts: wp.array2d[wp.float64],
    rows: int,
):
    j, row, chunk, lane = wp.tid()
    value = wp.float64(0.0)
    for e in range(offsets[row] + chunk * 128 + lane, offsets[row + 1], parts.shape[1] * 128):
        value += values[e] * x[j, cols[e]]
    total = wp.tile_extract(wp.tile_sum(wp.tile(value)), 0)
    if lane == 0:
        parts[j * rows + row, chunk] = total


@wp.kernel
def rect_mv_finish(parts: wp.array2d[wp.float64], y: wp.array2d[wp.float64]):
    j, row, lane = wp.tid()
    value = wp.float64(0.0)
    for chunk in range(lane, parts.shape[1], 32):
        value += parts[j * y.shape[1] + row, chunk]
    total = wp.tile_extract(wp.tile_sum(wp.tile(value)), 0)
    if lane == 0:
        y[j, row] = total


class _LongRowProduct:
    """Distribute long rectangular CSR rows across blocks, without atomics."""

    def __init__(self, matrix, width):
        self.matrix = matrix
        chunks = min(256, max(1, (matrix.nnz + matrix.nrow * 1024 - 1) // (matrix.nrow * 1024)))
        self.parts = wp.empty((width * matrix.nrow, chunks), dtype=wp.float64, device=matrix.device)

    def apply(self, x, y):
        a = self.matrix
        wp.launch_tiled(
            rect_mv_partials,
            dim=(x.shape[0], a.nrow, self.parts.shape[1]),
            inputs=[a.offsets, a.columns, a.values, x, self.parts, a.nrow],
            block_dim=128,
            device=a.device,
        )
        wp.launch_tiled(
            rect_mv_finish, dim=y.shape, inputs=[self.parts, y], block_dim=32, device=a.device
        )


class _Augmented:
    def __init__(self, a):
        if a.scalar_type != wp.float64 or a.block_shape != (1, 1) or a.row_counts is not None:
            raise ValueError("SVD needs compact scalar FP64 CSR")
        self.a = a
        self.at = ws.bsr_transposed(a)
        self.n = sum(a.shape)
        self.device = a.device
        self.long_forward = _LongRowProduct(a, 1) if a.nnz / a.nrow > 4096 else None
        self.long_backward = (
            _LongRowProduct(self.at, 1) if self.at.nnz / self.at.nrow > 4096 else None
        )

    def apply(self, x, y):
        a, b = self.a, self.at
        if self.long_forward is not None or self.long_backward is not None:
            split = a.shape[0]
            for matrix, product, source, output in (
                (a, self.long_forward, x[:, split:], y[:, :split]),
                (b, self.long_backward, x[:, :split], y[:, split:]),
            ):
                if product is not None:
                    product.apply(source, output)
                else:
                    wp.launch_tiled(
                        rect_mv_cooperative,
                        dim=output.shape,
                        inputs=[
                            matrix.offsets,
                            matrix.columns,
                            matrix.values,
                            source,
                            output,
                        ],
                        block_dim=32,
                        device=self.device,
                    )
            return
        args = [a.offsets, a.columns, a.values, b.offsets, b.columns, b.values, x, y, a.shape[0]]
        # Tall or wide snapshots have long rows in one half of the augmented
        # operator. Distribute those dot products over a warp instead of a lane.
        if max(a.nnz / a.nrow, b.nnz / b.nrow) > 64:
            wp.launch_tiled(
                augmented_mv_cooperative,
                dim=x.shape,
                inputs=args,
                block_dim=32,
                device=self.device,
            )
        else:
            wp.launch(augmented_mv, x.shape, args, device=self.device)


class PartialSVD:
    """Largest k singular triplets of rectangular real FP64 CSR.

    Left/right vectors use row-major (k,nrow)/(k,ncol) storage. An augmented
    symmetric operator avoids squaring the condition number. Always inspect
    both left and right residuals; zero singular subspaces are not unique.
    """

    def __init__(self, matrix, k, ncv=None, tol=1.0e-10):
        if not 1 <= k <= min(matrix.shape):
            raise ValueError("require 1 <= k <= min(matrix.shape)")
        self.op = _Augmented(matrix)
        self.device = matrix.device
        self.k = k
        self.core = KrylovSchur(
            self.op, k, ncv=ncv or min(self.op.n, max(4 * k, 48)), which="LA", tol=tol
        )
        self.u = wp.empty((k, matrix.shape[0]), dtype=wp.float64, device=self.device)
        self.v = wp.empty((k, matrix.shape[1]), dtype=wp.float64, device=self.device)
        self.singular_values = wp.empty(k, dtype=wp.float64, device=self.device)
        self.dots = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.norm = wp.zeros((1, 1), dtype=wp.float64, device=self.device)
        self.av = wp.empty_like(self.u)
        self.atu = wp.empty_like(self.v)
        self.left_norms = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.right_norms = wp.zeros_like(self.left_norms)
        self.left_residuals = wp.empty(k, dtype=wp.float64, device=self.device)
        self.right_residuals = wp.empty_like(self.left_residuals)
        self.order = wp.empty(k, dtype=wp.int32, device=self.device)
        self.count_left = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.count_right = wp.zeros_like(self.count_left)
        self.value_scratch = wp.empty_like(self.singular_values)
        self.tol = tol
        self.final_products = (
            _LongRowProduct(self.op.a, k) if self.op.long_forward is not None else None,
            _LongRowProduct(self.op.at, k) if self.op.long_backward is not None else None,
        )

    def finalize(self):
        wp.launch(
            split_vectors,
            self.core.eigenvectors.shape,
            [
                self.core.eigenvectors,
                self.u,
                self.v,
                self.core.eigenvalues,
                self.singular_values,
                self.u.shape[1],
            ],
            device=self.device,
        )
        for vectors in (self.u, self.v):
            n = vectors.shape[1]
            for j in range(self.k):
                if j:
                    for _ in range(2):
                        self.dots.zero_()
                        wp.launch_tiled(
                            K.basis_dots,
                            dim=(j, (n + 255) // 256),
                            inputs=[vectors, self.dots, j],
                            block_dim=128,
                            device=self.device,
                        )
                        wp.launch(K.basis_subtract, n, [vectors, self.dots, j], device=self.device)
                self.norm.zero_()
                wp.launch_tiled(
                    K.basis_norm,
                    dim=(n + 255) // 256,
                    inputs=[vectors, self.norm, j],
                    block_dim=128,
                    device=self.device,
                )
                wp.launch(K.basis_scale, n, [vectors, self.norm, j], device=self.device)
        a, b = self.op.a, self.op.at
        for matrix, product, source, output in (
            (a, self.final_products[0], self.v, self.av),
            (b, self.final_products[1], self.u, self.atu),
        ):
            if product is not None:
                product.apply(source, output)
            elif matrix.nnz / matrix.nrow > 64:
                wp.launch_tiled(
                    rect_mv_cooperative,
                    dim=output.shape,
                    inputs=[matrix.offsets, matrix.columns, matrix.values, source, output],
                    block_dim=32,
                    device=self.device,
                )
            else:
                wp.launch(
                    rect_mv,
                    output.shape,
                    [matrix.offsets, matrix.columns, matrix.values, source, output],
                    device=self.device,
                )
        wp.launch(K.identity_order, self.k, [self.order], device=self.device)
        for vec, prod, norms, errors, count in (
            (self.u, self.av, self.left_norms, self.left_residuals, self.count_left),
            (self.v, self.atu, self.right_norms, self.right_residuals, self.count_right),
        ):
            norms.zero_()
            count.zero_()
            wp.launch_tiled(
                K.residual,
                dim=(self.k, (vec.shape[1] + 255) // 256),
                inputs=[vec, prod, self.singular_values, self.order, norms],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                K.output_values,
                self.k,
                [
                    self.singular_values,
                    self.order,
                    self.value_scratch,
                    norms,
                    errors,
                    self.tol,
                    count,
                ],
                device=self.device,
            )

    def capture(self, max_iterations=100):
        self.graph = self.core.capture(max_iterations, finalizer=self.finalize)
        return self.graph

    def solve(self, iterations=30):
        self.core.solve(iterations)
        self.finalize()
        return self
