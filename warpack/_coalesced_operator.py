"""Sparse matrix application with adjacent RHS storage."""

import functools

import warp as wp

from ._coalesced_fsai import csr_eight_rhs, transpose_axpby, transpose_rows
from ._scratch import scratch


@functools.lru_cache(None)
def bsr_eight_rhs(dtype):
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
        rhs, slot = group * 8 + lane % 8, lane // 8
        value = vector(dtype(0.0))
        if rhs < x.shape[1]:
            for e in range(offsets[row] + slot, offsets[row + 1], 4):
                col = columns[e] * 3
                value += values[e] * vector(x[col, rhs], x[col + 1, rhs], x[col + 2, rhs])
        a = wp.tile_sum(wp.tile_reshape(wp.tile(value[0]), shape=(4, 8)), axis=0)
        b = wp.tile_sum(wp.tile_reshape(wp.tile(value[1]), shape=(4, 8)), axis=0)
        c = wp.tile_sum(wp.tile_reshape(wp.tile(value[2]), shape=(4, 8)), axis=0)
        if rhs < x.shape[1] and slot == 0:
            y[row * 3, rhs] = wp.tile_extract(a, lane % 8)
            y[row * 3 + 1, rhs] = wp.tile_extract(b, lane % 8)
            y[row * 3 + 2, rhs] = wp.tile_extract(c, lane % 8)

    return compute


class CoalescedSparseApplication:
    def __init__(self, matrix, width, _storage=None):
        if matrix.block_shape not in ((1, 1), (3, 3)) or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("require square scalar or 3x3 BSR matrix")
        if matrix.row_counts is not None:
            raise ValueError("require compressed sparse rows")
        self.matrix, self.width = matrix, width
        self.n, self.device = matrix.shape[0], matrix.device
        self.first, self.second = (
            scratch(_storage, name, (self.n, width), matrix.scalar_type, matrix.device)
            for name in ("matrix_first", "matrix_second")
        )

    def matvec(self, x, y, z, alpha, beta):
        wp.launch_tiled(
            transpose_rows,
            dim=((self.width + 31) // 32, (self.n + 31) // 32),
            inputs=[x.reshape((self.width, self.n)), self.first],
            block_dim=128,
            device=self.device,
        )
        a = self.matrix
        kernel = csr_eight_rhs if a.block_shape == (1, 1) else bsr_eight_rhs(a.scalar_type)
        wp.launch_tiled(
            kernel,
            dim=((self.width + 7) // 8, a.nrow),
            inputs=[a.offsets, a.columns, a.values, self.first, self.second],
            block_dim=32,
            device=self.device,
        )
        wp.launch_tiled(
            transpose_axpby,
            dim=((self.n + 31) // 32, (self.width + 31) // 32),
            inputs=[
                self.second,
                y.reshape((self.width, self.n)),
                z.reshape((self.width, self.n)),
                wp.float64(alpha),
                wp.float64(beta),
            ],
            block_dim=128,
            device=self.device,
        )
