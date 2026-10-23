"""Paired output reuse for ordinary and compact graphs, with immutable round data."""

import warp as wp

from warpack import kernels as K
from warpack._compact_jacobi import CompactJacobi, dynamic_pairs
from warpack.projected import GlobalJacobi


@wp.func
def paired_entry(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    partners: wp.array[wp.int32],
    i: int,
    j: int,
):
    n = h.shape[0]
    # Both pair kernels pair the final index with the current round index.
    # partners is immutable throughout this update, unlike round_index when
    # the compact variant advances its loop in thread(0,0).
    r = partners[n - 1]
    p, q = (r + i) % (n - 1), (r - i + n - 1) % (n - 1)
    u, w = (r + j) % (n - 1), (r - j + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    if j == 0:
        w = n - 1
    a00 = h[p, u]
    a01 = h[p, w]
    a10 = h[q, u]
    a11 = h[q, w]
    b00 = c[u] * a00 - s[u] * a01
    b01 = c[w] * a01 - s[w] * a00
    b10 = c[u] * a10 - s[u] * a11
    b11 = c[w] * a11 - s[w] * a10
    out_h[p, u] = c[p] * b00 - s[p] * b10
    out_h[p, w] = c[p] * b01 - s[p] * b11
    out_h[q, u] = c[q] * b10 - s[q] * b00
    out_h[q, w] = c[q] * b11 - s[q] * b01
    v00 = v[p, u]
    v01 = v[p, w]
    v10 = v[q, u]
    v11 = v[q, w]
    out_v[p, u] = c[u] * v00 - s[u] * v01
    out_v[p, w] = c[w] * v01 - s[w] * v00
    out_v[q, u] = c[u] * v10 - s[u] * v11
    out_v[q, w] = c[w] * v11 - s[w] * v10


@wp.kernel(enable_backward=False)
def update(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    partners: wp.array[wp.int32],
):
    i, j = wp.tid()
    paired_entry(h, v, out_h, out_v, c, s, partners, i, j)


@wp.kernel(enable_backward=False)
def advance(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    partners: wp.array[wp.int32],
    round_index: wp.array[wp.int32],
    running: wp.array[wp.int32],
    stop: int,
):
    i, j = wp.tid()
    paired_entry(h, v, out_h, out_v, c, s, partners, i, j)
    if i == 0 and j == 0:
        next_round = round_index[0] + 2
        round_index[0] = next_round
        running[0] = int(next_round < stop)


class RuntimePairedJacobi(GlobalJacobi):
    def __init__(self, *args):
        super().__init__(*args)
        if self.n < 2 or self.n > 1024 or self.n % 2:
            raise ValueError("even dimension in2..1024 required")

    def _sweep(self):
        current_h, current_v = self.h, self.v
        next_h, next_v = self.temp, self.vtemp
        for r in range(self.n - 1):
            wp.launch(
                K.jacobi_pairs,
                self.n // 2,
                [current_h, self.cs, self.ss, self.partners, r],
                device=self.device,
            )
            wp.launch(
                update,
                (self.n // 2, self.n // 2),
                [current_h, current_v, next_h, next_v, self.cs, self.ss, self.partners],
                block_dim=64,
                device=self.device,
            )
            current_h, next_h = next_h, current_h
            current_v, next_v = next_v, current_v
        wp.copy(self.h, current_h)
        wp.copy(self.v, current_v)
        self._finish_sweep()


class RuntimePairedCompact(CompactJacobi):
    def _round_pair(self):
        wp.launch(
            dynamic_pairs,
            self.n // 2,
            [self.h, self.cs, self.ss, self.partners, self.round_index, 0],
            device=self.device,
        )
        wp.launch(
            update,
            (self.n // 2, self.n // 2),
            [self.h, self.v, self.temp, self.vtemp, self.cs, self.ss, self.partners],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            dynamic_pairs,
            self.n // 2,
            [self.temp, self.cs, self.ss, self.partners, self.round_index, 1],
            device=self.device,
        )
        wp.launch(
            advance,
            (self.n // 2, self.n // 2),
            [
                self.temp,
                self.vtemp,
                self.h,
                self.v,
                self.cs,
                self.ss,
                self.partners,
                self.round_index,
                self.round_running,
                self.loop_rounds,
            ],
            block_dim=64,
            device=self.device,
        )

    def _sweep(self):
        if self.loop_rounds:
            self.round_index.zero_()
            self.round_running.fill_(1)
            wp.capture_while(self.round_running, self._round_chunk)
        current_h, current_v = self.h, self.v
        next_h, next_v = self.temp, self.vtemp
        for r in range(self.loop_rounds, self.n - 1):
            wp.launch(
                K.jacobi_pairs,
                self.n // 2,
                [current_h, self.cs, self.ss, self.partners, r],
                device=self.device,
            )
            wp.launch(
                update,
                (self.n // 2, self.n // 2),
                [current_h, current_v, next_h, next_v, self.cs, self.ss, self.partners],
                block_dim=64,
                device=self.device,
            )
            current_h, next_h = next_h, current_h
            current_v, next_v = next_v, current_v
        wp.copy(self.h, current_h)
        wp.copy(self.v, current_v)
        self._finish_sweep()
