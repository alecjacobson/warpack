"""Complex Hermitian problems through realification, with device basis recovery.

A complex n-by-n Hermitian matrix corresponds to the real symmetric operator
[[Re(A), -Im(A)], [Im(A), Re(A)]]. Each eigenvalue occurs twice; device-side
complex Gram-Schmidt removes those duplicate directions, including degeneracies.
"""

import warp as wp

from .solver import SymmetricEigensolver


@wp.func
def cmul(a: wp.vec2d, b: wp.vec2d):
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def cdot(a: wp.vec2d, b: wp.vec2d):
    return wp.vec2d(a[0] * b[0] + a[1] * b[1], a[0] * b[1] - a[1] * b[0])


@wp.func
def norm2(a: wp.vec2d):
    return wp.dot(a, a)


@wp.kernel
def realified_mv(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.vec2d],
    x: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
    n: int,
):
    j, i = wp.tid()
    val = wp.vec2d(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        c = cols[e]
        val += cmul(values[e], wp.vec2d(x[j, c], x[j, c + n]))
    out[j, i] = val[0]
    out[j, i + n] = val[1]


@wp.kernel
def candidate(real: wp.array2d[wp.float64], q: wp.array[wp.vec2d], j: int, n: int):
    i = wp.tid()
    q[i] = wp.vec2d(real[j, i], real[j, i + n])


@wp.kernel
def complex_dots(v: wp.array2d[wp.vec2d], q: wp.array[wp.vec2d], dots: wp.array2d[wp.vec2d]):
    j, chunk = wp.tid()
    a = wp.tile_load(v[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(q, shape=256, offset=chunk * 256)
    wp.tile_atomic_add(dots[j], wp.tile_sum(wp.tile_map(cdot, a, b)))


@wp.kernel
def complex_subtract(
    v: wp.array2d[wp.vec2d],
    q: wp.array[wp.vec2d],
    dots: wp.array2d[wp.vec2d],
    count: wp.array[wp.int32],
):
    i = wp.tid()
    val = q[i]
    for j in range(count[0]):
        val -= cmul(v[j, i], dots[j, 0])
    q[i] = val


@wp.kernel
def complex_norm(q: wp.array[wp.vec2d], norm: wp.array[wp.float64]):
    chunk = wp.tid()
    a = wp.tile_load(q, shape=256, offset=chunk * 256)
    wp.tile_atomic_add(norm, wp.tile_sum(wp.tile_map(norm2, a)))


@wp.kernel
def accept_vector(
    q: wp.array[wp.vec2d],
    v: wp.array2d[wp.vec2d],
    norm: wp.array[wp.float64],
    count: wp.array[wp.int32],
):
    i = wp.tid()
    if norm[0] > wp.float64(1.0e-12) and count[0] < v.shape[0]:
        v[count[0], i] = q[i] / wp.sqrt(norm[0])


@wp.kernel
def accept_value(
    values: wp.array[wp.float64],
    out: wp.array[wp.float64],
    norm: wp.array[wp.float64],
    count: wp.array[wp.int32],
    j: int,
):
    if norm[0] > wp.float64(1.0e-12) and count[0] < out.shape[0]:
        out[count[0]] = values[j]
        count[0] += 1


class HermitianCSR:
    """Device complex CSR; values use wp.vec2d(real, imaginary)."""

    def __init__(self, n, offsets, columns, values):
        if values.dtype != wp.vec2d or offsets.dtype != wp.int32 or columns.dtype != wp.int32:
            raise TypeError("expected int32 CSR indices and vec2d complex values")
        if len(offsets) != n + 1 or len(columns) != len(values):
            raise ValueError("CSR storage shape mismatch")
        if offsets.device != values.device or columns.device != values.device:
            raise ValueError("CSR arrays must be on the same device")
        self.complex_n = n
        self.n = 2 * n
        self.device = values.device
        self.offsets, self.columns, self.values = offsets, columns, values

    def apply(self, x, y):
        wp.launch(
            realified_mv,
            (x.shape[0], self.complex_n),
            [self.offsets, self.columns, self.values, x, y, self.complex_n],
            device=self.device,
        )


@wp.kernel
def row_bound(offsets: wp.array[wp.int32], values: wp.array[wp.vec2d], bound: wp.array[wp.float64]):
    i = wp.tid()
    val = wp.float64(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        val += wp.sqrt(wp.dot(values[e], values[e]))
    wp.atomic_max(bound, 0, val)


@wp.kernel
def shifted_real(
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    bound: wp.array[wp.float64],
    sign: float,
):
    j, i = wp.tid()
    y[j, i] = bound[0] * x[j, i] + wp.float64(sign) * y[j, i]


class _ExtremalIteration:
    def __init__(self, op, sign):
        self.op, self.sign = op, sign
        self.bound = wp.zeros(1, dtype=wp.float64, device=op.device)

    def apply(self, x, y):
        self.op.apply(x, y)
        self.bound.zero_()
        wp.launch(
            row_bound,
            self.op.complex_n,
            [self.op.offsets, self.op.values, self.bound],
            device=self.op.device,
        )
        wp.launch(shifted_real, x.shape, [x, y, self.bound, self.sign], device=self.op.device)


class HermitianEigensolver:
    def __init__(self, operator, k, ncv=None, which="LM", tol=1.0e-9):
        if not 1 <= k < operator.complex_n:
            raise ValueError("require 1 <= k < n")
        self.n, self.k, self.device = operator.complex_n, k, operator.device
        self.real = SymmetricEigensolver(
            operator,
            2 * k,
            ncv=ncv or min(2 * self.n, max(4 * k + 16, 32)),
            which=which,
            tol=tol,
            iteration_operator=(
                _ExtremalIteration(operator, -1.0 if which == "SA" else 1.0)
                if which in ("SA", "LA")
                else operator
            ),
        )
        self.eigenvectors = wp.zeros((k, self.n), dtype=wp.vec2d, device=self.device)
        self.eigenvalues = wp.zeros(k, dtype=wp.float64, device=self.device)
        self.q = wp.empty(self.n, dtype=wp.vec2d, device=self.device)
        self.dots = wp.zeros((k, 1), dtype=wp.vec2d, device=self.device)
        self.norm = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def finalize(self):
        self.eigenvectors.zero_()
        self.count.zero_()
        self.eigenvalues.zero_()
        for j in range(2 * self.k):
            wp.launch(
                candidate,
                self.n,
                [self.real.eigenvectors, self.q, j, self.n],
                device=self.device,
            )
            for _ in range(2):
                self.dots.zero_()
                wp.launch_tiled(
                    complex_dots,
                    dim=(self.k, (self.n + 255) // 256),
                    inputs=[self.eigenvectors, self.q, self.dots],
                    block_dim=128,
                    device=self.device,
                )
                wp.launch(
                    complex_subtract,
                    self.n,
                    [self.eigenvectors, self.q, self.dots, self.count],
                    device=self.device,
                )
            self.norm.zero_()
            wp.launch_tiled(
                complex_norm,
                dim=(self.n + 255) // 256,
                inputs=[self.q, self.norm],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                accept_vector,
                self.n,
                [self.q, self.eigenvectors, self.norm, self.count],
                device=self.device,
            )
            wp.launch(
                accept_value,
                1,
                [self.real.eigenvalues, self.eigenvalues, self.norm, self.count, j],
                device=self.device,
            )

    def solve(self, iterations=50):
        self.real.solve(iterations)
        self.finalize()
        return self

    def capture(self, max_iterations=100):
        self.graph = self.real.capture(max_iterations, finalizer=self.finalize)
        return self.graph
