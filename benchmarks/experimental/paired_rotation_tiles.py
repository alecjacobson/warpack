"""Benchmark-only FP64 paired Ritz-rotation tiles (#186)."""

import functools

import warp as wp

from warpack import BlockLOBPCG
from warpack.block_products import paired_ritz_rotation


@functools.lru_cache(None)
def rotation_kernel(rows, columns, inner):
    if rows not in (4, 8, 16) or columns not in (32, 64) or inner not in (8, 16, 32):
        raise ValueError("require output rows 4/8/16, columns 32/64 and inner tile 8/16/32")
    nr, nc, nk = wp.constant(rows), wp.constant(columns), wp.constant(inner)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        sx: wp.array2d[wp.float64],
        su: wp.array2d[wp.float64],
        cx: wp.array2d[wp.float64],
        cu: wp.array2d[wp.float64],
        x: wp.array2d[wp.float64],
        p: wp.array2d[wp.float64],
    ):
        i, j = wp.tid()
        acc_x = wp.tile_zeros(shape=(nr, nc), dtype=wp.float64)
        acc_u = wp.tile_zeros(shape=(nr, nc), dtype=wp.float64)
        k = int(0)
        while k < sx.shape[0]:
            a = wp.tile_load(cx, shape=(nr, nk), offset=(i * nr, k))
            b = wp.tile_load(sx, shape=(nk, nc), offset=(k, j * nc))
            wp.tile_matmul(a, b, acc_x)
            k += nk
        k = int(0)
        while k < su.shape[0]:
            a = wp.tile_load(cu, shape=(nr, nk), offset=(i * nr, k))
            b = wp.tile_load(su, shape=(nk, nc), offset=(k, j * nc))
            wp.tile_matmul(a, b, acc_u)
            k += nk
        wp.tile_store(x, acc_x + acc_u, offset=(i * nr, j * nc))
        wp.tile_store(p, acc_u, offset=(i * nr, j * nc))

    return compute


def rotation(sx, su, cx, cu, x, p, *, rows=16, columns=32, inner=32, threads=128):
    if threads not in (64, 128, 256):
        raise ValueError("require 64/128/256 threads")
    if (
        sx.shape[1] != su.shape[1]
        or cx.shape != (x.shape[0], sx.shape[0])
        or cu.shape != (x.shape[0], su.shape[0])
        or x.shape != (cx.shape[0], sx.shape[1])
        or p.shape != x.shape
    ):
        raise ValueError("incompatible paired-rotation shapes")
    kernel = rotation_kernel(rows, columns, inner)
    wp.launch_tiled(
        kernel,
        dim=((x.shape[0] + rows - 1) // rows, (x.shape[1] + columns - 1) // columns),
        inputs=[sx, su, cx, cu, x, p],
        block_dim=threads,
        device=x.device,
    )


class PairedRotationBlock(BlockLOBPCG):
    _rotation_rows = 16
    _rotation_columns = 32
    _rotation_inner = 32
    _rotation_threads = 128

    def rayleigh_ritz(self):
        original = wp.launch_tiled

        def launch(kernel, *args, **kwargs):
            if kernel is paired_ritz_rotation:
                output = kwargs["inputs"][4]
                kwargs = kwargs | dict(
                    dim=(
                        (output.shape[0] + self._rotation_rows - 1) // self._rotation_rows,
                        (output.shape[1] + self._rotation_columns - 1) // self._rotation_columns,
                    ),
                    block_dim=self._rotation_threads,
                )
                kernel = rotation_kernel(
                    self._rotation_rows, self._rotation_columns, self._rotation_inner
                )
            return original(kernel, *args, **kwargs)

        try:
            wp.launch_tiled = launch
            super().rayleigh_ritz()
        finally:
            wp.launch_tiled = original
