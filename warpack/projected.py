"""One-block cyclic Jacobi in shared memory, expressed entirely in Warp tiles."""

import functools

import warp as wp


@wp.func
def pair_left(i: int, r: int, n: int):
    return (r + i) % (n - 1)


@wp.func
def pair_right(i: int, r: int, n: int):
    q = (r - i + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    return q


@functools.lru_cache(None)
def jacobi_kernel(size):
    n = wp.constant(size)
    half = wp.constant(size // 2)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(a: wp.array2d[wp.float64], vectors: wp.array2d[wp.float64]):
        block, lane = wp.tid()
        h = wp.tile_load(a, shape=(n, n), storage="shared")
        v = wp.tile_zeros(shape=(n, n), dtype=wp.float64, storage="shared")
        cc = wp.tile_ones(shape=n, dtype=wp.float64, storage="shared")
        ss = wp.tile_zeros(shape=n, dtype=wp.float64, storage="shared")
        wp.tile_scatter_masked(
            v, wp.min(lane, n - 1), wp.min(lane, n - 1), wp.float64(1.0), lane < n
        )
        for sweep in range(10):
            for r in range(n - 1):
                ip = wp.min(lane, half - 1)
                p = pair_left(ip, r, n)
                q = pair_right(ip, r, n)
                apq = wp.tile_extract(h, p, q)
                app = wp.tile_extract(h, p, p)
                aqq = wp.tile_extract(h, q, q)
                c = wp.float64(1.0)
                s = wp.float64(0.0)
                if wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(app) + wp.abs(aqq)):
                    tau = (aqq - app) / (wp.float64(2.0) * apq)
                    t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                    if tau < wp.float64(0.0):
                        t = -t
                    c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                    s = t * c
                wp.tile_scatter_masked(cc, ip, c, lane < half)
                wp.tile_scatter_masked(ss, ip, s, lane < half)
                for batch in range((half * half + 255) // 256):
                    ix = batch * 256 + lane
                    safe = wp.min(ix, half * half - 1)
                    i = safe // half
                    j = safe % half
                    p = pair_left(i, r, n)
                    q = pair_right(i, r, n)
                    u = pair_left(j, r, n)
                    w = pair_right(j, r, n)
                    c = wp.tile_extract(cc, i)
                    s = wp.tile_extract(ss, i)
                    d = wp.tile_extract(cc, j)
                    t = wp.tile_extract(ss, j)
                    a00 = wp.tile_extract(h, p, u)
                    a01 = wp.tile_extract(h, p, w)
                    a10 = wp.tile_extract(h, q, u)
                    a11 = wp.tile_extract(h, q, w)
                    b00 = c * a00 - s * a10
                    b01 = c * a01 - s * a11
                    b10 = s * a00 + c * a10
                    b11 = s * a01 + c * a11
                    wp.tile_scatter_masked(h, p, u, d * b00 - t * b01, ix < half * half)
                    wp.tile_scatter_masked(h, p, w, t * b00 + d * b01, ix < half * half)
                    wp.tile_scatter_masked(h, q, u, d * b10 - t * b11, ix < half * half)
                    wp.tile_scatter_masked(h, q, w, t * b10 + d * b11, ix < half * half)
                for batch in range((n * half + 255) // 256):
                    ix = batch * 256 + lane
                    safe = wp.min(ix, n * half - 1)
                    i = safe // half
                    j = safe % half
                    p = pair_left(j, r, n)
                    q = pair_right(j, r, n)
                    c = wp.tile_extract(cc, j)
                    s = wp.tile_extract(ss, j)
                    a0 = wp.tile_extract(v, i, p)
                    a1 = wp.tile_extract(v, i, q)
                    wp.tile_scatter_masked(v, i, p, c * a0 - s * a1, ix < n * half)
                    wp.tile_scatter_masked(v, i, q, s * a0 + c * a1, ix < n * half)
        wp.tile_store(a, h)
        wp.tile_store(vectors, v)

    return compute
