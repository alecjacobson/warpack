"""Shared FP32 complete-basis initializer; final recovery remains FP64.

Derived from Warp-only shared projected Jacobi with a deliberately coarse
2e-5 stopping bound. No FP32 eigenvalues or ranks leave this initializer.
"""

import functools

import warp as wp

from benchmarks.experimental.fp32_jacobi import FP32Jacobi


@wp.func
def pair_left32(i: int, r: int, n: int):
    return (r + i) % (n - 1)


@wp.func
def pair_right32(i: int, r: int, n: int):
    q = (r - i + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    return q


@functools.lru_cache(None)
def shared_jacobi32_kernel(size, block_dim=256):
    threads = wp.constant(block_dim)
    n = wp.constant(size)
    half = wp.constant(size // 2)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        a: wp.array2d[wp.float32], vectors: wp.array2d[wp.float32], sweeps: wp.array[wp.int32]
    ):
        block, lane = wp.tid()
        h = wp.tile_load(a, shape=(n, n), storage="shared")
        v = wp.tile_zeros(shape=(n, n), dtype=wp.float32, storage="shared")
        cc = wp.tile_ones(shape=n, dtype=wp.float32, storage="shared")
        ss = wp.tile_zeros(shape=n, dtype=wp.float32, storage="shared")
        wp.tile_scatter_masked(
            v, wp.min(lane, n - 1), wp.min(lane, n - 1), wp.float32(1.0), lane < n
        )
        for sweep in range(30):
            for r in range(n - 1):
                ip = wp.min(lane, half - 1)
                p = pair_left32(ip, r, n)
                q = pair_right32(ip, r, n)
                apq = wp.tile_extract(h, p, q)
                app = wp.tile_extract(h, p, p)
                aqq = wp.tile_extract(h, q, q)
                c = wp.float32(1.0)
                s = wp.float32(0.0)
                if lane < half and wp.abs(apq) > wp.float32(1.0e-7) * (wp.abs(app) + wp.abs(aqq)):
                    tau = (aqq - app) / (wp.float32(2.0) * apq)
                    t = wp.float32(1.0) / (wp.abs(tau) + wp.sqrt(wp.float32(1.0) + tau * tau))
                    if tau < wp.float32(0.0):
                        t = -t
                    c = wp.float32(1.0) / wp.sqrt(wp.float32(1.0) + t * t)
                    s = t * c
                wp.tile_scatter_masked(cc, ip, c, lane < half)
                wp.tile_scatter_masked(ss, ip, s, lane < half)
                for batch in range((half * half + threads - 1) // threads):
                    ix = batch * threads + lane
                    safe = wp.min(ix, half * half - 1)
                    i = safe // half
                    j = safe % half
                    p = pair_left32(i, r, n)
                    q = pair_right32(i, r, n)
                    u = pair_left32(j, r, n)
                    w = pair_right32(j, r, n)
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
                for batch in range((n * half + threads - 1) // threads):
                    ix = batch * threads + lane
                    safe = wp.min(ix, n * half - 1)
                    i = safe // half
                    j = safe % half
                    p = pair_left32(j, r, n)
                    q = pair_right32(j, r, n)
                    c = wp.tile_extract(cc, j)
                    s = wp.tile_extract(ss, j)
                    a0 = wp.tile_extract(v, i, p)
                    a1 = wp.tile_extract(v, i, q)
                    wp.tile_scatter_masked(v, i, p, c * a0 - s * a1, ix < n * half)
                    wp.tile_scatter_masked(v, i, q, s * a0 + c * a1, ix < n * half)
            if lane == 0:
                sweeps[0] = sweep + 1
            off = wp.float32(0.0)
            diagonal = wp.float32(0.0)
            for batch in range((n * n + threads - 1) // threads):
                ix = batch * threads + lane
                i, j = wp.min(ix // n, n - 1), ix % n
                value = wp.abs(wp.tile_extract(h, i, j))
                if ix < n * n:
                    if i == j:
                        diagonal = wp.max(diagonal, value)
                    else:
                        off = wp.max(off, value)
            max_off = wp.tile_max(wp.tile(off))
            max_diagonal = wp.tile_max(wp.tile(diagonal))
            if wp.tile_extract(max_off, 0) <= wp.float32(2.0e-5) * wp.tile_extract(max_diagonal, 0):
                break
        wp.tile_store(a, h)
        wp.tile_store(vectors, v)

    return compute


def shared_bytes(n):
    return 4 * (2 * n * n + 2 * n) + 2048


class SharedFP32Jacobi(FP32Jacobi):
    """Use the shared initializer only when its complete workspaces fit."""

    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        if not h.device.is_cuda or shared_bytes(h.shape[0]) > h.device.max_shared_memory_per_block:
            raise ValueError("FP32 shared initializer exceeds CUDA shared-memory capacity")
        super().__init__(h, v, cs, ss, partners, temp, vtemp)

    def apply(self):
        wp.launch_tiled(
            shared_jacobi32_kernel(self.h.shape[0]),
            dim=1,
            inputs=[self.h, self.v, self.sweeps],
            block_dim=256,
            device=self.device,
        )
