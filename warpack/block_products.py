"""FP64 Gram and basis-update kernels for the preconditioned block solver."""

import functools

import warp as wp


@wp.kernel
def initial_row_norms(x: wp.array2d[wp.float64], norms: wp.array[wp.float64]):
    row, chunk = wp.tid()
    values = wp.tile_load(x[row], shape=256, offset=chunk * 256)
    total = wp.tile_sum(values * values)
    wp.atomic_add(norms, row, wp.tile_extract(total, 0))


@wp.kernel
def perturb_initial_rows(
    initial: wp.array2d[wp.float64],
    norms: wp.array[wp.float64],
    random_rows: wp.array2d[wp.float64],
    amount: wp.float64,
):
    i, j = wp.tid()
    scale = amount * wp.sqrt(norms[i] / wp.float64(initial.shape[1]))
    random_rows[i, j] = initial[i, j] + scale * random_rows[i, j]


@functools.lru_cache(None)
def gram_kernel(chunk):
    blocks = wp.constant(chunk // 32)
    stride = wp.constant(chunk)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(x: wp.array2d[wp.float64], y: wp.array2d[wp.float64], g: wp.array2d[wp.float64]):
        i, j, piece = wp.tid()
        acc = wp.tile_zeros(shape=(8, 8), dtype=wp.float64)
        for k in range(blocks):
            a = wp.tile_load(x, shape=(8, 32), offset=(i * 8, piece * stride + k * 32))
            b = wp.tile_load(y, shape=(8, 32), offset=(j * 8, piece * stride + k * 32))
            wp.tile_matmul(a, wp.tile_transpose(b), acc)
        wp.tile_atomic_add(g, acc, offset=(i * 8, j * 8))

    return compute


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


@wp.kernel(enable_backward=False)
def wide_paired_ritz_rotation(
    sx: wp.array2d[wp.float64],
    su: wp.array2d[wp.float64],
    cx: wp.array2d[wp.float64],
    cu: wp.array2d[wp.float64],
    x: wp.array2d[wp.float64],
    p: wp.array2d[wp.float64],
):
    # Eight output rows reuse each basis tile across more products. Keep the
    # original inner width of eight, FP64, separate X/U views and accumulators.
    i, j = wp.tid()
    acc_x = wp.tile_zeros(shape=(8, 32), dtype=wp.float64)
    acc_u = wp.tile_zeros(shape=(8, 32), dtype=wp.float64)
    k = int(0)
    while k < sx.shape[0]:
        a = wp.tile_load(cx, shape=(8, 8), offset=(i * 8, k))
        b = wp.tile_load(sx, shape=(8, 32), offset=(k, j * 32))
        wp.tile_matmul(a, b, acc_x)
        k += 8
    k = int(0)
    while k < su.shape[0]:
        a = wp.tile_load(cu, shape=(8, 8), offset=(i * 8, k))
        b = wp.tile_load(su, shape=(8, 32), offset=(k, j * 32))
        wp.tile_matmul(a, b, acc_u)
        k += 8
    wp.tile_store(x, acc_x + acc_u, offset=(i * 8, j * 32))
    wp.tile_store(p, acc_u, offset=(i * 8, j * 32))


def _use_wide_paired_rotation(solver):
    """Static capture configuration; no device values or solver state are read."""
    return (
        getattr(solver, "_wide_paired_rotation_enabled", True)
        and solver.device.is_cuda
        and solver.device.arch == 89
        and 24 <= solver.b <= 256
        and solver.n >= getattr(solver, "_wide_paired_rotation_minimum", 32768)
    )


@wp.kernel(enable_backward=False)
def wide_subtract_projection(
    q: wp.array2d[wp.float64],
    coeff: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(8, 32), dtype=wp.float64)
    for block in range((q.shape[0] + 7) // 8):
        a = wp.tile_load(coeff, shape=(8, 8), offset=(i * 8, block * 8))
        b = wp.tile_load(q, shape=(8, 32), offset=(block * 8, j * 32))
        wp.tile_matmul(a, b, acc)
    old = wp.tile_load(out, shape=(8, 32), offset=(i * 8, j * 32))
    wp.tile_store(out, old - acc, offset=(i * 8, j * 32))


def _use_wide_projection(solver):
    """Static capture configuration; preserve the original unqualified path."""
    return (
        getattr(solver, "_wide_projection_enabled", True)
        and solver.device.is_cuda
        and solver.device.arch == 89
        and 24 <= solver.b <= 256
        and solver.n >= getattr(solver, "_wide_projection_minimum", 32768)
    )
