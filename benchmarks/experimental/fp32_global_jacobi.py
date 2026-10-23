"""Multiblock FP32 initializer derived from the global FP64 Jacobi schedule.

Only supplies a complete starting basis; original-matrix FP64 recovery is
required before using any values or rank decisions. Benchmark-only.
"""

import warp as wp

from benchmarks.experimental.fp32_jacobi import FP32Jacobi


@wp.kernel
def jacobi_init(h: wp.array2d[wp.float32], v: wp.array2d[wp.float32]):
    i, j = wp.tid()
    v[i, j] = wp.float32(0.0)
    if i == j:
        v[i, j] = wp.float32(1.0)
    if i < j:
        a = (h[i, j] + h[j, i]) * wp.float32(0.5)
        h[i, j] = a
        h[j, i] = a


@wp.kernel
def jacobi_pairs(
    h: wp.array2d[wp.float32],
    cs: wp.array[wp.float32],
    ss: wp.array[wp.float32],
    partners: wp.array[wp.int32],
    round: int,
):
    i = wp.tid()
    n = h.shape[0]
    p = (round + i) % (n - 1)
    q = (round - i + n - 1) % (n - 1)
    if i == 0:
        q = n - 1
    apq = h[p, q]
    c = wp.float32(1.0)
    s = wp.float32(0.0)
    if wp.abs(apq) > wp.float32(1.0e-7) * (wp.abs(h[p, p]) + wp.abs(h[q, q])):
        tau = (h[q, q] - h[p, p]) / (wp.float32(2.0) * apq)
        t = wp.float32(1.0) / (wp.abs(tau) + wp.sqrt(wp.float32(1.0) + tau * tau))
        if tau < wp.float32(0.0):
            t = -t
        c = wp.float32(1.0) / wp.sqrt(wp.float32(1.0) + t * t)
        s = t * c
    cs[p] = c
    cs[q] = c
    ss[p] = s
    ss[q] = -s
    partners[p] = q
    partners[q] = p


@wp.kernel
def jacobi_two_sided(
    h: wp.array2d[wp.float32],
    v: wp.array2d[wp.float32],
    out_h: wp.array2d[wp.float32],
    out_v: wp.array2d[wp.float32],
    c: wp.array[wp.float32],
    s: wp.array[wp.float32],
    p: wp.array[wp.int32],
):
    # Out-of-place J^T H J and V J for disjoint plane rotations. Preserve the
    # right-then-left arithmetic order without materializing the first product.
    i, j = wp.tid()
    right_i = c[j] * h[i, j] - s[j] * h[i, p[j]]
    right_p = c[j] * h[p[i], j] - s[j] * h[p[i], p[j]]
    out_h[i, j] = c[i] * right_i - s[i] * right_p
    out_v[i, j] = c[j] * v[i, j] - s[j] * v[i, p[j]]


@wp.kernel
def global_jacobi_update(
    h: wp.array2d[wp.float32],
    sweeps: wp.array[wp.int32],
    running: wp.array[wp.int32],
):
    block, lane = wp.tid()
    n = h.shape[0]
    off = wp.float32(0.0)
    diagonal = wp.float32(0.0)
    for entry in range(lane, n * n, 128):
        i, j = entry // n, entry % n
        value = wp.abs(h[i, j])
        if not wp.isfinite(value):
            off = wp.float32(wp.inf)
        elif i == j:
            diagonal = wp.max(diagonal, value)
        else:
            off = wp.max(off, value)
    max_off = wp.tile_extract(wp.tile_max(wp.tile(off)), 0)
    max_diagonal = wp.tile_extract(wp.tile_max(wp.tile(diagonal)), 0)
    if lane == 0:
        sweeps[0] += 1
        running[0] = 0
        if sweeps[0] < 30 and (
            not wp.isfinite(max_off) or max_off > wp.float32(2.0e-5) * max_diagonal
        ):
            running[0] = 1


class GlobalFP32Jacobi(FP32Jacobi):
    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        super().__init__(h, v, cs, ss, partners, temp, vtemp)
        self.n = h.shape[0]
        self.running = wp.zeros_like(self.sweeps)
        self.graph = None

    def _sweep(self):
        current_h, current_v = self.h, self.v
        next_h, next_v = self.temp, self.vtemp
        for r in range(self.n - 1):
            wp.launch(
                jacobi_pairs,
                self.n // 2,
                [current_h, self.cs, self.ss, self.partners, r],
                device=self.device,
            )
            wp.launch(
                jacobi_two_sided,
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

    def _capture(self):
        wp.launch(jacobi_init, self.h.shape, [self.h, self.v], device=self.device)
        self.sweeps.zero_()
        self.running.fill_(1)
        wp.capture_while(self.running, self._sweep)

    def apply(self):
        if wp.get_stream(self.device).is_capturing:
            self._capture()
            return
        if self.graph is None:
            wp.load_module(module=jacobi_init.module, device=self.device)
            wp.load_module(module=global_jacobi_update.module, device=self.device, block_dim=128)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._capture()
            self.graph = capture.graph
        wp.capture_launch(self.graph)
