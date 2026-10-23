"""Compact device round loops for global Jacobi graph construction.

Same FP64 pair order and round arithmetic as GlobalJacobi. Even groups restore
ping-pong buffers; trailing rounds remain explicit. This optional path trades
graph construction cost for additional device-loop overhead.
"""

import warp as wp

from . import kernels as K
from .projected import GlobalJacobi, global_jacobi_update


@wp.kernel
def dynamic_pairs(
    h: wp.array2d[wp.float64],
    cs: wp.array[wp.float64],
    ss: wp.array[wp.float64],
    partners: wp.array[wp.int32],
    round_index: wp.array[wp.int32],
    offset: int,
):
    i = wp.tid()
    round = round_index[0] + offset
    n = h.shape[0]
    p = (round + i) % (n - 1)
    q = (round - i + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    apq = h[p, q]
    c = wp.float64(1.0)
    s = wp.float64(0.0)
    if wp.abs(apq) > wp.float64(1.0e-16) * (wp.abs(h[p, p]) + wp.abs(h[q, q])):
        tau = (h[q, q] - h[p, p]) / (wp.float64(2.0) * apq)
        t = wp.float64(1.0) / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
        if tau < wp.float64(0.0):
            t = -t
        c = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
        s = t * c
    cs[p] = c
    cs[q] = c
    ss[p] = s
    ss[q] = -s
    partners[p] = q
    partners[q] = p


@wp.kernel
def two_sided_advance(
    h: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    out_h: wp.array2d[wp.float64],
    out_v: wp.array2d[wp.float64],
    c: wp.array[wp.float64],
    s: wp.array[wp.float64],
    p: wp.array[wp.int32],
    round_index: wp.array[wp.int32],
    running: wp.array[wp.int32],
    round_stop: int,
):
    # Out-of-place J^T H J and V J for disjoint plane rotations. Preserve the
    # right-then-left arithmetic order without materializing the first product.
    i, j = wp.tid()
    right_i = c[j] * h[i, j] - s[j] * h[i, p[j]]
    right_p = c[j] * h[p[i], j] - s[j] * h[p[i], p[j]]
    out_h[i, j] = c[i] * right_i - s[i] * right_p
    out_v[i, j] = c[j] * v[i, j] - s[j] * v[i, p[j]]
    if i == 0 and j == 0:
        next_round = round_index[0] + 2
        round_index[0] = next_round
        running[0] = int(next_round < round_stop)


class CompactJacobi(GlobalJacobi):
    def __init__(self, h, v, cs, ss, partners, temp, vtemp, rounds_per_loop=2):
        n = h.shape[0]
        if not h.device.is_cuda or n < 2 or n > 1024 or n % 2:
            raise ValueError("compact Jacobi requires CUDA and an even dimension in 2..1024")
        if rounds_per_loop not in (2, 8, 16, 32):
            raise ValueError("rounds_per_loop must be 2, 8, 16 or 32")
        self.rounds_per_loop = rounds_per_loop
        self.loop_rounds = ((n - 1) // rounds_per_loop) * rounds_per_loop
        super().__init__(h, v, cs, ss, partners, temp, vtemp)
        self.round_index = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.round_running = wp.zeros_like(self.round_index)
        wp.load_module(module=dynamic_pairs.module, device=self.device)

    def _round_pair(self):
        wp.launch(
            dynamic_pairs,
            self.n // 2,
            [self.h, self.cs, self.ss, self.partners, self.round_index, 0],
            device=self.device,
        )
        wp.launch(
            K.jacobi_two_sided,
            self.h.shape,
            [self.h, self.v, self.temp, self.vtemp, self.cs, self.ss, self.partners],
            device=self.device,
        )
        wp.launch(
            dynamic_pairs,
            self.n // 2,
            [self.temp, self.cs, self.ss, self.partners, self.round_index, 1],
            device=self.device,
        )
        wp.launch(
            two_sided_advance,
            self.h.shape,
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
            device=self.device,
        )

    def _round_chunk(self):
        for _ in range(self.rounds_per_loop // 2):
            self._round_pair()

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
                K.jacobi_two_sided,
                self.h.shape,
                [current_h, current_v, next_h, next_v, self.cs, self.ss, self.partners],
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
