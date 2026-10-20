"""Cooperative projected complex QR prototype; no vendor dense eigensolver.

Householder similarity, shifts, deflation and back substitution follow the
serial reference. Kernel boundaries order reflector sides; each cooperative
QR kernel keeps row/column ownership fixed through its Givens sequence.
"""

import operator
from functools import lru_cache

import warp as wp

from warpack.hermitian import cdot, cmul, norm2
from warpack.schur import cabs, cdiv, conj, csqrt


@wp.func
def lower_part(i: int, k: int, value: wp.vec2d):
    result = wp.vec2d(0.0)
    if i > k:
        result = value
    return result


@wp.func
def add_at(i: int, k: int, value: wp.vec2d, bump: wp.vec2d):
    result = value
    if i == k:
        result += bump
    return result


@wp.func
def clear_at(i: int, k: int, value: wp.vec2d):
    result = value
    if i == k:
        result = wp.vec2d(0.0)
    return result


@wp.func
def complex_scale(value: wp.vec2d, scale: wp.float64):
    return value * scale


@wp.kernel
def initialize(
    h: wp.array2d[wp.vec2d],
    q: wp.array2d[wp.vec2d],
    state: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    i, j = wp.tid()
    q[i, j] = wp.vec2d(0.0)
    if i == j:
        q[i, j] = wp.vec2d(1.0, 0.0)
    if i == 0 and j == 0:
        state[0] = h.shape[0] - 1
        state[1] = 0
        status[0] = 0


@wp.kernel
def update_left(
    h: wp.array2d[wp.vec2d],
    u: wp.array[wp.vec2d],
    coefficients: wp.array2d[wp.vec2d],
    enabled: wp.array[wp.int32],
    k: int,
):
    i, j = wp.tid()
    if enabled[0] != 0 and j >= k:
        h[i, j] -= wp.float64(2.0) * cmul(u[i], coefficients[0, j])


@wp.kernel
def update_right(
    h: wp.array2d[wp.vec2d],
    q: wp.array2d[wp.vec2d],
    u: wp.array[wp.vec2d],
    coefficients: wp.array2d[wp.vec2d],
    enabled: wp.array[wp.int32],
    k: int,
):
    i, j = wp.tid()
    if enabled[0] != 0:
        h[i, j] -= wp.float64(2.0) * cmul(coefficients[1, i], conj(u[j]))
        q[i, j] -= wp.float64(2.0) * cmul(coefficients[2, i], conj(u[j]))
        if j == k and i > k + 1:
            h[i, j] = wp.vec2d(0.0)


@wp.kernel
def advance(
    h: wp.array2d[wp.vec2d],
    state: wp.array[wp.int32],
    status: wp.array[wp.int32],
    running: wp.array[wp.int32],
    shift: wp.array[wp.vec2d],
    maxiter: int,
    increment: int,
):
    state[1] += increment
    end = state[0]
    while end > 0:
        threshold = wp.float64(2e-15) * (
            cabs(h[end - 1, end - 1]) + cabs(h[end, end])
        ) + wp.float64(1e-300)
        if cabs(h[end, end - 1]) <= threshold:
            h[end, end - 1] = wp.vec2d(0.0)
            end -= 1
        else:
            break
    state[0] = end
    running[0] = 0
    if end > 0:
        if state[1] >= maxiter:
            status[0] = 2
        else:
            a, b = h[end - 1, end - 1], h[end - 1, end]
            c, d = h[end, end - 1], h[end, end]
            delta = (a - d) * wp.float64(0.5)
            root = csqrt(cmul(delta, delta) + cmul(b, c))
            mu = (a + d) * wp.float64(0.5) + root
            other = (a + d) * wp.float64(0.5) - root
            if cabs(other - d) < cabs(mu - d):
                mu = other
            if state[1] % 40 == 39:
                mu = d + wp.vec2d(cabs(c) * wp.float64(0.75), 0.0)
            shift[0] = mu
            if wp.isfinite(mu[0]) and wp.isfinite(mu[1]):
                running[0] = 1
            else:
                status[0] = 2


@wp.kernel
def shift_diagonal(
    h: wp.array2d[wp.vec2d],
    state: wp.array[wp.int32],
    shift: wp.array[wp.vec2d],
    sign: wp.float64,
):
    i = wp.tid()
    if i <= state[0]:
        h[i, i] += sign * shift[0]


@wp.kernel
def diagonal_values(
    h: wp.array2d[wp.vec2d], values: wp.array[wp.vec2d], scale: wp.array[wp.float64]
):
    largest = wp.float64(0.0)
    for i in range(h.shape[0]):
        values[i] = h[i, i]
        largest = wp.max(largest, cabs(h[i, i]))
    scale[0] = largest


@wp.kernel
def triangular_vectors(
    h: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    scale: wp.array[wp.float64],
    out: wp.array2d[wp.vec2d],
):
    j = wp.tid()
    for i in range(h.shape[0]):
        out[i, j] = wp.vec2d(0.0)
    out[j, j] = wp.vec2d(1.0, 0.0)
    for step in range(j):
        i = j - 1 - step
        value = wp.vec2d(0.0)
        for k in range(i + 1, j + 1):
            value += cmul(h[i, k], out[k, j])
        den = h[i, i] - values[j]
        if cabs(den) < wp.float64(1e-14) * scale[0]:
            den = wp.vec2d(wp.max(wp.float64(1e-14) * scale[0], wp.float64(1e-300)), 0.0)
        out[i, j] = -cdiv(value, den)


@wp.kernel
def multiply_vectors(
    q: wp.array2d[wp.vec2d], triangular: wp.array2d[wp.vec2d], out: wp.array2d[wp.vec2d]
):
    i, j = wp.tid()
    value = wp.vec2d(0.0)
    for k in range(q.shape[0]):
        value += cmul(q[i, k], triangular[k, j])
    out[i, j] = value


@lru_cache(None)
def kernels(size):
    width = wp.constant(size)

    @wp.kernel(module="unique")
    def householder(
        h: wp.array2d[wp.vec2d],
        u: wp.array[wp.vec2d],
        enabled: wp.array[wp.int32],
        k: int,
    ):
        indices = wp.tile_arange(0, width, dtype=wp.int32)
        column = wp.tile_reshape(wp.tile_load(h, shape=(width, 1), offset=(0, k)), shape=(width,))
        masked = wp.tile_map(
            lower_part, indices, wp.tile_full(shape=width, value=k, dtype=wp.int32), column
        )
        radius = wp.sqrt(wp.tile_extract(wp.tile_sum(wp.tile_map(norm2, masked)), 0))
        wp.tile_store(
            enabled, wp.tile_full(shape=1, value=int(radius > wp.float64(1e-150)), dtype=wp.int32)
        )
        if radius > wp.float64(1e-150):
            a = h[k + 1, k]
            phase = wp.vec2d(1.0, 0.0)
            if cabs(a) > wp.float64(1e-150):
                phase = a / cabs(a)
            uu = wp.tile_map(
                add_at,
                indices,
                wp.tile_full(shape=width, value=k + 1, dtype=wp.int32),
                masked,
                wp.tile_full(shape=width, value=phase * radius, dtype=wp.vec2d),
            )
            length = wp.sqrt(wp.tile_extract(wp.tile_sum(wp.tile_map(norm2, uu)), 0))
            normalized = wp.tile_map(
                complex_scale,
                uu,
                wp.tile_full(shape=width, value=wp.float64(1.0) / length, dtype=wp.float64),
            )
            wp.tile_store(u, normalized)

    @wp.kernel(module="unique")
    def left_coefficients(
        h: wp.array2d[wp.vec2d],
        u: wp.array[wp.vec2d],
        coefficients: wp.array2d[wp.vec2d],
        enabled: wp.array[wp.int32],
    ):
        j = wp.tid()
        if enabled[0] != 0:
            uu = wp.tile_load(u, shape=width)
            column = wp.tile_reshape(
                wp.tile_load(h, shape=(width, 1), offset=(0, j)), shape=(width,)
            )
            value = wp.tile_sum(wp.tile_map(cdot, uu, column))
            wp.tile_store(coefficients[0], value, offset=j)

    @wp.kernel(module="unique")
    def right_coefficients(
        h: wp.array2d[wp.vec2d],
        q: wp.array2d[wp.vec2d],
        u: wp.array[wp.vec2d],
        coefficients: wp.array2d[wp.vec2d],
        enabled: wp.array[wp.int32],
    ):
        i = wp.tid()
        if enabled[0] != 0:
            uu = wp.tile_load(u, shape=width)
            hr = wp.tile_load(h[i], shape=width)
            qr = wp.tile_load(q[i], shape=width)
            wp.tile_store(coefficients[1], wp.tile_sum(wp.tile_map(cmul, hr, uu)), offset=i)
            wp.tile_store(coefficients[2], wp.tile_sum(wp.tile_map(cmul, qr, uu)), offset=i)

    @wp.kernel(module="unique")
    def qr_factor(
        h: wp.array2d[wp.vec2d],
        cs: wp.array[wp.float64],
        ss: wp.array[wp.vec2d],
        state: wp.array[wp.int32],
    ):
        indices = wp.tile_arange(0, width, dtype=wp.int32)
        for i in range(state[0]):
            x = wp.tile_load(h[i], shape=width)
            y = wp.tile_load(h[i + 1], shape=width)
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
            wp.tile_store(h[i], nx)
            wp.tile_store(h[i + 1], ny)
            wp.tile_store(cs, wp.tile_full(shape=1, value=co, dtype=wp.float64), offset=i)
            wp.tile_store(ss, wp.tile_full(shape=1, value=si, dtype=wp.vec2d), offset=i)

    @wp.kernel(module="unique")
    def rq_update(
        h: wp.array2d[wp.vec2d],
        q: wp.array2d[wp.vec2d],
        cs: wp.array[wp.float64],
        ss: wp.array[wp.vec2d],
        state: wp.array[wp.int32],
    ):
        for j in range(state[0]):
            ct = wp.tile_full(shape=width, value=cs[j], dtype=wp.float64)
            st = wp.tile_full(shape=width, value=ss[j], dtype=wp.vec2d)
            sc = wp.tile_full(shape=width, value=conj(ss[j]), dtype=wp.vec2d)
            hx = wp.tile_reshape(wp.tile_load(h, shape=(width, 1), offset=(0, j)), shape=(width,))
            hy = wp.tile_reshape(
                wp.tile_load(h, shape=(width, 1), offset=(0, j + 1)), shape=(width,)
            )
            qx = wp.tile_reshape(wp.tile_load(q, shape=(width, 1), offset=(0, j)), shape=(width,))
            qy = wp.tile_reshape(
                wp.tile_load(q, shape=(width, 1), offset=(0, j + 1)), shape=(width,)
            )
            nhx = wp.tile_map(complex_scale, hx, ct) + wp.tile_map(cmul, sc, hy)
            nhy = -wp.tile_map(cmul, st, hx) + wp.tile_map(complex_scale, hy, ct)
            nqx = wp.tile_map(complex_scale, qx, ct) + wp.tile_map(cmul, sc, qy)
            nqy = -wp.tile_map(cmul, st, qx) + wp.tile_map(complex_scale, qy, ct)
            wp.tile_store(h, wp.tile_reshape(nhx, shape=(width, 1)), offset=(0, j))
            wp.tile_store(h, wp.tile_reshape(nhy, shape=(width, 1)), offset=(0, j + 1))
            wp.tile_store(q, wp.tile_reshape(nqx, shape=(width, 1)), offset=(0, j))
            wp.tile_store(q, wp.tile_reshape(nqy, shape=(width, 1)), offset=(0, j + 1))

    @wp.kernel(module="unique")
    def normalize(vectors: wp.array2d[wp.vec2d]):
        j = wp.tid()
        column = wp.tile_reshape(
            wp.tile_load(vectors, shape=(width, 1), offset=(0, j)), shape=(width,)
        )
        radius = wp.sqrt(
            wp.max(wp.tile_extract(wp.tile_sum(wp.tile_map(norm2, column)), 0), wp.float64(1e-300))
        )
        result = wp.tile_map(
            complex_scale,
            column,
            wp.tile_full(shape=width, value=wp.float64(1.0) / radius, dtype=wp.float64),
        )
        wp.tile_store(vectors, wp.tile_reshape(result, shape=(width, 1)), offset=(0, j))

    return householder, left_coefficients, right_coefficients, qr_factor, rq_update, normalize


class TiledSchur:
    """Fixed workspace, device-controlled QR, reusable inside outer capture."""

    def __init__(self, h, q, vectors, values, u, cs, ss, status, maxiter=None):
        self.n, self.device = h.shape[0], h.device
        if h.shape != (self.n, self.n) or h.dtype != wp.vec2d or not self.device.is_cuda:
            raise ValueError("require a square complex128 CUDA projection")
        if not 1 <= self.n <= 64:
            raise ValueError("prototype supports projection sizes 1 through 64")
        for buffer, shape, dtype in (
            (q, h.shape, wp.vec2d),
            (vectors, h.shape, wp.vec2d),
            (values, (self.n,), wp.vec2d),
            (u, (self.n,), wp.vec2d),
            (cs, (self.n,), wp.float64),
            (ss, (self.n,), wp.vec2d),
            (status, (1,), wp.int32),
        ):
            if buffer.shape != shape or buffer.dtype != dtype or buffer.device != self.device:
                raise ValueError("Schur workspace shape, dtype or device mismatch")
        self.h, self.q, self.vectors, self.values = h, q, vectors, values
        self.u, self.cs, self.ss, self.status = u, cs, ss, status
        self.maxiter = self.n * 100 if maxiter is None else operator.index(maxiter)
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        self.state = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.running = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.enabled = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.shift = wp.zeros(1, dtype=wp.vec2d, device=self.device)
        self.scale = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.coefficients = wp.empty((3, self.n), dtype=wp.vec2d, device=self.device)
        self.triangular = wp.empty_like(h)
        self.width = 1 << (self.n - 1).bit_length()
        self.block_dim = max(32, self.width)
        self.householder, self.left, self.right, self.factor, self.rq, self.normalize = kernels(
            self.width
        )
        wp.load_module(module=initialize.module, device=self.device)
        for kernel in (
            self.householder,
            self.left,
            self.right,
            self.factor,
            self.rq,
            self.normalize,
        ):
            wp.load_module(module=kernel.module, device=self.device, block_dim=self.block_dim)
        # The supported 570 driver needs pending setup writes completed before
        # graph upload; this is setup-only, as in the production capture paths.
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.capture_while(self.running, self.qr_step)
        self.eager_qr = capture.graph

    def launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim, inputs, device=self.device)

    def tiled(self, kernel, dim, inputs):
        wp.launch_tiled(
            kernel, dim=dim, inputs=inputs, block_dim=self.block_dim, device=self.device
        )

    def qr_step(self):
        self.launch(shift_diagonal, self.n, [self.h, self.state, self.shift, -1.0])
        self.tiled(self.factor, 1, [self.h, self.cs, self.ss, self.state])
        self.tiled(self.rq, 1, [self.h, self.q, self.cs, self.ss, self.state])
        self.launch(shift_diagonal, self.n, [self.h, self.state, self.shift, 1.0])
        self.launch(
            advance, 1, [self.h, self.state, self.status, self.running, self.shift, self.maxiter, 1]
        )

    def run(self):
        self.launch(initialize, (self.n, self.n), [self.h, self.q, self.state, self.status])
        for k in range(self.n - 2):
            self.tiled(self.householder, 1, [self.h, self.u, self.enabled, k])
            self.tiled(self.left, self.n, [self.h, self.u, self.coefficients, self.enabled])
            self.launch(
                update_left, (self.n, self.n), [self.h, self.u, self.coefficients, self.enabled, k]
            )
            self.tiled(
                self.right, self.n, [self.h, self.q, self.u, self.coefficients, self.enabled]
            )
            self.launch(
                update_right,
                (self.n, self.n),
                [self.h, self.q, self.u, self.coefficients, self.enabled, k],
            )
        self.launch(
            advance, 1, [self.h, self.state, self.status, self.running, self.shift, self.maxiter, 0]
        )
        if wp.get_stream(self.device).is_capturing:
            wp.capture_while(self.running, self.qr_step)
        else:
            wp.capture_launch(self.eager_qr)
        self.launch(diagonal_values, 1, [self.h, self.values, self.scale])
        self.launch(triangular_vectors, self.n, [self.h, self.values, self.scale, self.triangular])
        self.launch(multiply_vectors, (self.n, self.n), [self.q, self.triangular, self.vectors])
        self.tiled(self.normalize, self.n, [self.vectors])
