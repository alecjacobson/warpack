"""One-launch Jacobi rounds with recomputed angles; no production dispatch change."""

import functools

import warp as wp

from warpack.projected import GlobalJacobi, global_jacobi_update


@wp.func
def angle(h: wp.array2d[wp.float64], index: int, round: int):
    n = h.shape[0]
    partner = (2 * round - index + 2 * (n - 1)) % (n - 1)
    left = (index - round + n - 1) % (n - 1) < n // 2
    if index == n - 1:
        partner = round
        left = False
    elif index == round:
        partner = n - 1
        left = True
    p, q = index, partner
    if not left:
        p, q = partner, index
    apq, app, aqq = h[p, q], h[p, p], h[q, q]
    c, s = wp.float64(1.0), wp.float64(0.0)
    if wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(app) + wp.abs(aqq)):
        tau = (aqq - app) / (wp.float64(2.0) * apq)
        t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
        if tau < wp.float64(0.0):
            t = -t
        c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
        s = t * c
    if not left:
        s = -s
    return c, s, partner


@wp.kernel(enable_backward=False)
def scalar_round(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    round: int,
):
    i, j = wp.tid()
    ci, si, pi = angle(h, i, round)
    cj, sj, pj = angle(h, j, round)
    right_i = cj * h[i, j] - sj * h[i, pj]
    right_p = cj * h[pi, j] - sj * h[pi, pj]
    out_h[i, j] = ci * right_i - si * right_p
    out_v[i, j] = cj * v[i, j] - sj * v[i, pj]


@functools.lru_cache(None)
def tiled_round(size):
    tile_size = wp.constant(size)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        h: wp.array2d[wp.float64],
        v: wp.array2d[wp.float64],
        out_h: wp.array2d[wp.float64],
        out_v: wp.array2d[wp.float64],
        round: int,
    ):
        bi, bj, lane = wp.tid()
        rc = wp.tile_zeros(shape=tile_size, dtype=wp.float64, storage="shared")
        rs = wp.tile_zeros(shape=tile_size, dtype=wp.float64, storage="shared")
        rp = wp.tile_zeros(shape=tile_size, dtype=wp.int32, storage="shared")
        cc = wp.tile_zeros(shape=tile_size, dtype=wp.float64, storage="shared")
        cs = wp.tile_zeros(shape=tile_size, dtype=wp.float64, storage="shared")
        cp = wp.tile_zeros(shape=tile_size, dtype=wp.int32, storage="shared")
        ci, si, pi = wp.float64(1.0), wp.float64(0.0), int(0)
        cj, sj, pj = wp.float64(1.0), wp.float64(0.0), int(0)
        if lane < tile_size:
            ci, si, pi = angle(h, wp.min(bi * tile_size + lane, h.shape[0] - 1), round)
            cj, sj, pj = angle(h, wp.min(bj * tile_size + lane, h.shape[0] - 1), round)
        position = wp.min(lane, tile_size - 1)
        wp.tile_scatter_masked(rc, position, ci, lane < tile_size)
        wp.tile_scatter_masked(rs, position, si, lane < tile_size)
        wp.tile_scatter_masked(rp, position, pi, lane < tile_size)
        wp.tile_scatter_masked(cc, position, cj, lane < tile_size)
        wp.tile_scatter_masked(cs, position, sj, lane < tile_size)
        wp.tile_scatter_masked(cp, position, pj, lane < tile_size)
        for batch in range((tile_size * tile_size + 255) // 256):
            local = batch * 256 + lane
            ri, rj = local // tile_size, local % tile_size
            i, j = bi * tile_size + ri, bj * tile_size + rj
            ci, si, pi = wp.tile_extract(rc, ri), wp.tile_extract(rs, ri), wp.tile_extract(rp, ri)
            cj, sj, pj = wp.tile_extract(cc, rj), wp.tile_extract(cs, rj), wp.tile_extract(cp, rj)
            if i < h.shape[0] and j < h.shape[0]:
                right_i = cj * h[i, j] - sj * h[i, pj]
                right_p = cj * h[pi, j] - sj * h[pi, pj]
                out_h[i, j] = ci * right_i - si * right_p
                out_v[i, j] = cj * v[i, j] - sj * v[i, pj]

    return compute


class OnTheFlyJacobi(GlobalJacobi):
    def __init__(self, h, v, cs, ss, partners, temp, vtemp, *, variant="tile16"):
        if h.shape[0] % 2 or h.shape[0] < 2:
            raise ValueError("even dimension >=2 required")
        if variant not in ("scalar", "tile16", "tile32"):
            raise ValueError("unknown round variant")
        super().__init__(h, v, cs, ss, partners, temp, vtemp)
        self.variant = variant

    def _sweep(self):
        current_h, current_v = self.h, self.v
        next_h, next_v = self.temp, self.vtemp
        for r in range(self.n - 1):
            inputs = [current_h, current_v, next_h, next_v, r]
            if self.variant == "scalar":
                wp.launch(scalar_round, self.h.shape, inputs, device=self.device)
            else:
                size = 16 if self.variant == "tile16" else 32
                wp.launch_tiled(
                    tiled_round(size),
                    dim=((self.n + size - 1) // size,) * 2,
                    inputs=inputs,
                    block_dim=256,
                    device=self.device,
                )
            current_h, next_h = next_h, current_h
            current_v, next_v = next_v, current_v
        if current_h is not self.h:
            wp.copy(self.h, current_h)
            wp.copy(self.v, current_v)
        wp.launch_tiled(
            global_jacobi_update,
            dim=1,
            inputs=[self.h, self.sweeps, self.running],
            block_dim=128,
            device=self.device,
        )
