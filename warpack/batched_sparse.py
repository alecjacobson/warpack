"""Pure-Warp sparse products sharing matrix reads across FP64 right-hand sides."""

from typing import Any

import warp as wp

from ._validation import fp64_rows
from .solver import SparseOperator


@wp.kernel(module="unique")
def csr_four_rhs(
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


@wp.kernel(module="unique")
def csr_four_rhs_padded(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array(dtype=Any),
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = x.dtype(0.0)
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row] + counts[row], 8):
            value += x.dtype(values[e]) * x[rhs, columns[e]]
    totals = wp.tile_sum(wp.tile_reshape(wp.tile(value), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row] = wp.tile_extract(totals, lane // 8)


def csr_block_product(matrix, x, y):
    if matrix.row_counts is not None:
        wp.launch_tiled(
            csr_four_rhs_padded,
            dim=((x.shape[0] + 3) // 4, matrix.nrow),
            inputs=[matrix.offsets, matrix.row_counts, matrix.columns, matrix.values, x, y],
            block_dim=32,
            device=matrix.device,
        )
        return
    wp.launch_tiled(
        csr_four_rhs,
        dim=((x.shape[0] + 3) // 4, matrix.nrow),
        inputs=[matrix.offsets, matrix.columns, matrix.values, x, y],
        block_dim=32,
        device=matrix.device,
    )


@wp.kernel
def bsr_four_rhs(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    a: wp.array[wp.mat33d],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    group, row, lane = wp.tid()
    rhs, slot = group * 4 + lane // 8, lane % 8
    value = wp.vec3d(wp.float64(0.0))
    if rhs < x.shape[0]:
        for e in range(offsets[row] + slot, offsets[row + 1], 8):
            col = columns[e] * 3
            value += a[e] * wp.vec3d(x[rhs, col], x[rhs, col + 1], x[rhs, col + 2])
    tx = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=1)
    ty = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=1)
    tz = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=1)
    if rhs < x.shape[0] and slot == 0:
        y[rhs, row * 3] = wp.tile_extract(tx, lane // 8)
        y[rhs, row * 3 + 1] = wp.tile_extract(ty, lane // 8)
        y[rhs, row * 3 + 2] = wp.tile_extract(tz, lane // 8)


class BatchedSparseOperator(SparseOperator):
    """FP64 CSR/3×3 BSR products sharing reads across adjacent RHS rows.

    Same constructor/contract as SparseOperator; inputs and outputs are distinct
    (width, n) arrays; strided views are supported. Shifted/signed operators
    use the base product.
    Sparse topology remains fixed during graph replay; values may be updated.

    Args:
        matrix: Compressed square FP64 Warp BsrMatrix (scalar CSR or 3x3 BSR), on the solver
            device.
        shift: Add shift*x to each output row; this is an additive diagonal shift, not the
            shift-invert sigma convention.
        sign: Multiply the matrix product by this real scalar: y = sign*A*x + shift*x.

    Attributes:
        n: Scalar matrix dimension.
        device: Warp device inferred from matrix.
        matrix: Borrowed Warp sparse matrix; maintain its storage lifetime.
    """

    def apply(self, x, y):
        """Apply the operator to device rows, writing into the supplied output.

        Args:
            x: Input device rows of the dtype and shape required by this operator.
            y: Output device rows, matching x. Follow the class-specific aliasing rules; do not
                assume overlapping views are supported.

        x and y are FP64 arrays of shape (width, n). Strided views are supported; their storage must not overlap.

        Returns:
            None. Results are written to the documented device arrays.
        """
        fp64_rows(x, y, self.n, self.device)
        a = self.matrix
        if a.values.dtype == wp.float64 and self.shift == 0 and self.sign == 1:
            csr_block_product(a, x, y)
            return
        if a.values.dtype != wp.mat33d or self.shift != 0 or self.sign != 1:
            return super().apply(x, y)
        wp.launch_tiled(
            bsr_four_rhs,
            dim=((x.shape[0] + 3) // 4, a.nrow),
            inputs=[a.offsets, a.columns, a.values, x, y],
            block_dim=32,
            device=self.device,
        )
