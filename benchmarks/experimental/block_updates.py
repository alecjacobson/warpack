"""Optional FP64 block updates, kept outside the supported eigensolver API."""

import functools

import warp as wp


@functools.lru_cache(None)
def symmetric_gram_kernel(chunk):
    """Compute one triangle of X A X^T when A is known symmetric.

    The caller supplies Y = A X. Off-diagonal tile pairs share the same rounded
    product; diagonal tiles retain both dot products for later symmetrization.
    This is not valid for arbitrary rectangular cross-Gram products.
    """
    blocks = wp.constant(chunk // 32)
    stride = wp.constant(chunk)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(x: wp.array2d[wp.float64], y: wp.array2d[wp.float64], g: wp.array2d[wp.float64]):
        i, j, piece = wp.tid()
        if i <= j:
            acc = wp.tile_zeros(shape=(8, 8), dtype=wp.float64)
            for k in range(blocks):
                a = wp.tile_load(x, shape=(8, 32), offset=(i * 8, piece * stride + k * 32))
                b = wp.tile_load(y, shape=(8, 32), offset=(j * 8, piece * stride + k * 32))
                wp.tile_matmul(a, wp.tile_transpose(b), acc)
            wp.tile_atomic_add(g, acc, offset=(i * 8, j * 8))
            if i != j:
                wp.tile_atomic_add(g, wp.tile_transpose(acc), offset=(j * 8, i * 8))

    return compute


@wp.kernel
def subtract_projection(
    q: wp.array2d[wp.float64],
    coeff: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
):
    # out is disjoint from q/coeff. Each tile reads/writes its own out entries.
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(4, 32), dtype=wp.float64)
    for block in range((q.shape[0] + 7) // 8):
        a = wp.tile_load(coeff, shape=(4, 8), offset=(i * 4, block * 8))
        b = wp.tile_load(q, shape=(8, 32), offset=(block * 8, j * 32))
        wp.tile_matmul(a, b, acc)
    old = wp.tile_load(out, shape=(4, 32), offset=(i * 4, j * 32))
    wp.tile_store(out, old - acc, offset=(i * 4, j * 32))


@wp.kernel
def paired_ritz_rotation(
    sx: wp.array2d[wp.float64],
    su: wp.array2d[wp.float64],
    cx: wp.array2d[wp.float64],
    cu: wp.array2d[wp.float64],
    x: wp.array2d[wp.float64],
    p: wp.array2d[wp.float64],
):
    # X_new = Cx X_old + Cu U; P_new = Cu U. Reuse the shared product.
    # Separate bounded views keep partial inner tiles from crossing X/U.
    i, j = wp.tid()
    acc_x = wp.tile_zeros(shape=(4, 32), dtype=wp.float64)
    acc_u = wp.tile_zeros(shape=(4, 32), dtype=wp.float64)
    for block in range((sx.shape[0] + 7) // 8):
        a = wp.tile_load(cx, shape=(4, 8), offset=(i * 4, block * 8))
        b = wp.tile_load(sx, shape=(8, 32), offset=(block * 8, j * 32))
        wp.tile_matmul(a, b, acc_x)
    for block in range((su.shape[0] + 7) // 8):
        a = wp.tile_load(cu, shape=(4, 8), offset=(i * 4, block * 8))
        b = wp.tile_load(su, shape=(8, 32), offset=(block * 8, j * 32))
        wp.tile_matmul(a, b, acc_u)
    wp.tile_store(x, acc_x + acc_u, offset=(i * 4, j * 32))
    wp.tile_store(p, acc_u, offset=(i * 4, j * 32))
