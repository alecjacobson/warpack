"""Experimental block-product tuning; production defaults remain unchanged."""

import functools
from typing import Any

import warp as wp
import warp.sparse as ws

from warpack import SparseOperator
from warpack.block_products import gram_kernel as gram_kernel


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


def csr_block_product(matrix, x, y):
    wp.launch_tiled(
        csr_four_rhs,
        dim=((x.shape[0] + 3) // 4, matrix.nrow),
        inputs=[matrix.offsets, matrix.columns, matrix.values, x, y],
        block_dim=32,
        device=matrix.device,
    )


@functools.lru_cache(None)
def rotate_kernel(rows, columns, inner):
    m, n, k = wp.constant(rows), wp.constant(columns), wp.constant(inner)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        q: wp.array2d[wp.float64], coeff: wp.array2d[wp.float64], out: wp.array2d[wp.float64]
    ):
        i, j = wp.tid()
        acc = wp.tile_zeros(shape=(m, n), dtype=wp.float64)
        for block in range((q.shape[0] + k - 1) // k):
            a = wp.tile_load(coeff, shape=(m, k), offset=(i * m, block * k))
            b = wp.tile_load(q, shape=(k, n), offset=(block * k, j * n))
            wp.tile_matmul(a, b, acc)
        wp.tile_store(out, acc, offset=(i * m, j * n))

    return compute


class UpstreamSparseOperator(SparseOperator):
    """Evaluate each block row using upstream Warp's tiled sparse MV."""

    def apply(self, x, y):
        if self.shift != 0 or self.sign != 1:
            return super().apply(x, y)
        for i in range(x.shape[0]):
            ws.bsr_mv(self.matrix, x[i], y[i], tile_size=32)


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


class BatchedSparseOperator(UpstreamSparseOperator):
    def apply(self, x, y):
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
