"""FP64 Gram tiles for the qualified Ada block-solver path."""

import functools

import warp as wp


@functools.lru_cache(None)
def wide_gram_kernel(symmetric):
    extent, blocks, stride = wp.constant(16), wp.constant(128), wp.constant(4096)
    mirror = wp.constant(bool(symmetric))

    @wp.kernel(enable_backward=False, module="unique")
    def compute(x: wp.array2d[wp.float64], y: wp.array2d[wp.float64], out: wp.array2d[wp.float64]):
        i, j, piece = wp.tid()
        if not mirror or i <= j:
            acc = wp.tile_zeros(shape=(extent, extent), dtype=wp.float64)
            # A rolled loop reuses tile storage, including on older compilers.
            k = int(0)
            while k < blocks:
                a = wp.tile_load(
                    x, shape=(extent, 32), offset=(i * extent, piece * stride + k * 32)
                )
                b = wp.tile_load(
                    y, shape=(extent, 32), offset=(j * extent, piece * stride + k * 32)
                )
                wp.tile_matmul(a, wp.tile_transpose(b), acc)
                k += 1
            wp.tile_atomic_add(out, acc, offset=(i * extent, j * extent))
            if mirror and i != j:
                wp.tile_atomic_add(out, wp.tile_transpose(acc), offset=(j * extent, i * extent))

    return compute


def gram(x, y, out, symmetric=False):
    out.zero_()
    wp.launch_tiled(
        wide_gram_kernel(symmetric),
        dim=((x.shape[0] + 15) // 16, (y.shape[0] + 15) // 16, (x.shape[1] + 4095) // 4096),
        inputs=[x, y, out],
        block_dim=128,
        device=x.device,
    )
