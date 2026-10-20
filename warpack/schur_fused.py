"""Keep the Hessenberg QR loop in one cooperative Warp block.

H lives in a shared tile; column ownership stays fixed for global Schur-vector
updates. Shared-tile assignments provide synchronization between reflector
sides. No host decisions, conditional graph launches, or vendor dense solver.
"""

from functools import lru_cache

import warp as wp

from warpack.hermitian import cmul
from warpack.schur import cabs, conj, csqrt
from warpack.schur_tiled import (
    TiledSchur,
    clear_at,
    complex_scale,
)


@wp.func
def copy_value(value: wp.vec2d):
    return value


@lru_cache(None)
def qr_loop_kernel(size):
    width = wp.constant(size)
    entries = wp.constant(size * size)

    @wp.kernel(module="unique")
    def qr_loop(
        h: wp.array2d[wp.vec2d],
        q: wp.array2d[wp.vec2d],
        state: wp.array[wp.int32],
        status: wp.array[wp.int32],
        running: wp.array[wp.int32],
        maxiter: int,
    ):
        hh = wp.tile_load(h, shape=(width, width), storage="shared")
        flat = wp.tile_reshape(hh, shape=(entries,))
        cosines = wp.tile_zeros(shape=width, dtype=wp.float64, storage="shared")
        sines = wp.tile_zeros(shape=width, dtype=wp.vec2d, storage="shared")
        indices = wp.tile_arange(0, width, dtype=wp.int32)
        end = h.shape[0] - 1
        iteration = int(0)
        failed = int(0)
        while end > 0:
            threshold = wp.float64(2e-15) * (
                cabs(wp.tile_extract(flat, (end - 1) * width + (end - 1)))
                + cabs(wp.tile_extract(flat, (end) * width + (end)))
            ) + wp.float64(1e-300)
            if cabs(wp.tile_extract(flat, (end) * width + (end - 1))) <= threshold:
                flat[end * width + end - 1] = wp.vec2d(0.0)
                end -= 1
            else:
                if iteration >= maxiter:
                    failed = 2
                    break
                a = wp.tile_extract(flat, (end - 1) * width + (end - 1))
                b = wp.tile_extract(flat, (end - 1) * width + (end))
                c = wp.tile_extract(flat, (end) * width + (end - 1))
                d = wp.tile_extract(flat, (end) * width + (end))
                delta = (a - d) * wp.float64(0.5)
                root = csqrt(cmul(delta, delta) + cmul(b, c))
                mu = (a + d) * wp.float64(0.5) + root
                other = (a + d) * wp.float64(0.5) - root
                if cabs(other - d) < cabs(mu - d):
                    mu = other
                if iteration % 40 == 39:
                    mu = d + wp.vec2d(cabs(c) * wp.float64(0.75), 0.0)
                if not wp.isfinite(mu[0]) or not wp.isfinite(mu[1]):
                    failed = 2
                    break
                for i in range(end + 1):
                    flat[i * width + i] = wp.tile_extract(flat, (i) * width + (i)) - mu
                for i in range(end):
                    x = wp.tile_reshape(
                        wp.tile_view(hh, shape=(1, width), offset=(i, 0)), shape=(width,)
                    )
                    y = wp.tile_reshape(
                        wp.tile_view(hh, shape=(1, width), offset=(i + 1, 0)), shape=(width,)
                    )
                    a, b = wp.tile_extract(x, i), wp.tile_extract(y, i)
                    aa, bb = cabs(a), cabs(b)
                    radius = wp.sqrt(aa * aa + bb * bb)
                    co, si = wp.float64(1.0), wp.vec2d(0.0)
                    if radius > wp.float64(1e-150):
                        co = aa / radius
                        if aa > wp.float64(1e-150):
                            si = cmul(a / aa, conj(b)) / radius
                        else:
                            si = conj(b) / bb
                    ct = wp.tile_full(shape=width, value=co, dtype=wp.float64)
                    st = wp.tile_full(shape=width, value=si, dtype=wp.vec2d)
                    sc = wp.tile_full(shape=width, value=conj(si), dtype=wp.vec2d)
                    nx = wp.tile_map(complex_scale, x, ct) + wp.tile_map(cmul, st, y)
                    ny = -wp.tile_map(cmul, sc, x) + wp.tile_map(complex_scale, y, ct)
                    ny = wp.tile_map(
                        clear_at, indices, wp.tile_full(shape=width, value=i, dtype=wp.int32), ny
                    )
                    wp.tile_assign(hh, wp.tile_reshape(nx, shape=(1, width)), offset=(i, 0))
                    wp.tile_assign(hh, wp.tile_reshape(ny, shape=(1, width)), offset=(i + 1, 0))
                    cosines[i] = co
                    sines[i] = si
                for j in range(end):
                    co, si = wp.tile_extract(cosines, j), wp.tile_extract(sines, j)
                    ct = wp.tile_full(shape=width, value=co, dtype=wp.float64)
                    st = wp.tile_full(shape=width, value=si, dtype=wp.vec2d)
                    sc = wp.tile_full(shape=width, value=conj(si), dtype=wp.vec2d)
                    hx = wp.tile_reshape(
                        wp.tile_map(copy_value, wp.tile_view(hh, shape=(width, 1), offset=(0, j))),
                        shape=(width,),
                    )
                    hy = wp.tile_reshape(
                        wp.tile_map(
                            copy_value, wp.tile_view(hh, shape=(width, 1), offset=(0, j + 1))
                        ),
                        shape=(width,),
                    )
                    qx = wp.tile_reshape(
                        wp.tile_load(q, shape=(width, 1), offset=(0, j)), shape=(width,)
                    )
                    qy = wp.tile_reshape(
                        wp.tile_load(q, shape=(width, 1), offset=(0, j + 1)), shape=(width,)
                    )
                    nhx = wp.tile_map(complex_scale, hx, ct) + wp.tile_map(cmul, sc, hy)
                    nhy = -wp.tile_map(cmul, st, hx) + wp.tile_map(complex_scale, hy, ct)
                    nqx = wp.tile_map(complex_scale, qx, ct) + wp.tile_map(cmul, sc, qy)
                    nqy = -wp.tile_map(cmul, st, qx) + wp.tile_map(complex_scale, qy, ct)
                    wp.tile_assign(hh, wp.tile_reshape(nhx, shape=(width, 1)), offset=(0, j))
                    wp.tile_assign(hh, wp.tile_reshape(nhy, shape=(width, 1)), offset=(0, j + 1))
                    wp.tile_store(q, wp.tile_reshape(nqx, shape=(width, 1)), offset=(0, j))
                    wp.tile_store(q, wp.tile_reshape(nqy, shape=(width, 1)), offset=(0, j + 1))
                # The first shared assignment synchronizes the last global Q
                # writes before any later iteration changes row ownership.
                for i in range(end + 1):
                    flat[i * width + i] = wp.tile_extract(flat, (i) * width + (i)) + mu
                iteration += 1
        wp.tile_store(h, hh)
        wp.tile_store(state, wp.tile_full(shape=1, value=end, dtype=wp.int32), offset=0)
        wp.tile_store(state, wp.tile_full(shape=1, value=iteration, dtype=wp.int32), offset=1)
        wp.tile_store(status, wp.tile_full(shape=1, value=failed, dtype=wp.int32))
        wp.tile_store(running, wp.tile_zeros(shape=1, dtype=wp.int32))

    return qr_loop


def fused_supported(device, size):
    # Conservative headroom for Warp tile synchronization/reduction scratch.
    # A padded64x64 complex tile alone uses64KiB. Older GPUs fall back to
    # the existing tiled QR instead of failing during kernel compilation.
    if not device.is_cuda or not 1 <= size <= 64:
        return False
    width = 1 << (size - 1).bit_length()
    required = 96 * 1024 if width == 64 else 32 * 1024
    return device.max_shared_memory_per_block >= required


class FusedSchur(TiledSchur):
    def prepare_qr(self):
        if not fused_supported(self.device, self.n):
            raise ValueError("insufficient shared memory for fused Schur projection")
        self.fused_qr = qr_loop_kernel(self.width)
        wp.load_module(module=self.fused_qr.module, device=self.device, block_dim=self.block_dim)

    def run_qr(self):
        self.tiled(
            self.fused_qr, 1, [self.h, self.q, self.state, self.status, self.running, self.maxiter]
        )
