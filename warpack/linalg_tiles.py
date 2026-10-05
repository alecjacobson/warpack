"""Tiled FP64 dense products, implemented with Warp tile primitives."""

import warp as wp


@wp.kernel
def rotate_tiled(
    q: wp.array2d[wp.float64],
    coeff: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(4, 32), dtype=wp.float64)
    for block in range((q.shape[0] + 7) // 8):
        a = wp.tile_load(coeff, shape=(4, 8), offset=(i * 4, block * 8))
        b = wp.tile_load(q, shape=(8, 32), offset=(block * 8, j * 32))
        wp.tile_matmul(a, b, acc)
    wp.tile_store(out, acc, offset=(i * 4, j * 32))


@wp.kernel
def gram_tiled(x: wp.array2d[wp.float64], y: wp.array2d[wp.float64], g: wp.array2d[wp.float64]):
    i, j, chunk = wp.tid()
    acc = wp.tile_zeros(shape=(8, 8), dtype=wp.float64)
    for k in range(8):
        a = wp.tile_load(x, shape=(8, 32), offset=(i * 8, chunk * 256 + k * 32))
        b = wp.tile_load(y, shape=(8, 32), offset=(j * 8, chunk * 256 + k * 32))
        wp.tile_matmul(a, wp.tile_transpose(b), acc)
    wp.tile_atomic_add(g, acc, offset=(i * 8, j * 8))
