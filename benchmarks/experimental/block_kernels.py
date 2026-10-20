"""Benchmark-only alternate products; supported batched products live in warpack."""

import functools

import warp as wp
import warp.sparse as ws

from warpack import SparseOperator
from warpack.batched_sparse import BatchedSparseOperator as BatchedSparseOperator
from warpack.batched_sparse import csr_block_product as csr_block_product
from warpack.block_products import gram_kernel as gram_kernel


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
