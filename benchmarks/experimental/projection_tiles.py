"""Benchmark-only output-tile controls for FP64 projection subtraction (#192)."""

import functools

import warp as wp

from warpack import BlockLOBPCG


@functools.lru_cache(None)
def projection_kernel(rows):
    if rows not in (4, 8, 16):
        raise ValueError("require 4, 8 or 16 output rows")
    nr = wp.constant(rows)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        q: wp.array2d[wp.float64],
        coeff: wp.array2d[wp.float64],
        out: wp.array2d[wp.float64],
    ):
        i, j = wp.tid()
        acc = wp.tile_zeros(shape=(nr, 32), dtype=wp.float64)
        for block in range((q.shape[0] + 7) // 8):
            a = wp.tile_load(coeff, shape=(nr, 8), offset=(i * nr, block * 8))
            b = wp.tile_load(q, shape=(8, 32), offset=(block * 8, j * 32))
            wp.tile_matmul(a, b, acc)
        old = wp.tile_load(out, shape=(nr, 32), offset=(i * nr, j * 32))
        wp.tile_store(out, old - acc, offset=(i * nr, j * 32))

    return compute


def subtract(q, coeff, out, *, rows=8):
    if q.shape[1] != out.shape[1] or coeff.shape != (out.shape[0], q.shape[0]):
        raise ValueError("incompatible projection shapes")
    wp.launch_tiled(
        projection_kernel(rows),
        dim=((out.shape[0] + rows - 1) // rows, (out.shape[1] + 31) // 32),
        inputs=[q, coeff, out],
        block_dim=128,
        device=out.device,
    )


class ProjectionBlock(BlockLOBPCG):
    _projection_rows = 8

    def subtract(self, basis, coeff, target, scratch):
        if self.fused_updates:
            subtract(basis, coeff, target, rows=self._projection_rows)
        else:
            super().subtract(basis, coeff, target, scratch)
