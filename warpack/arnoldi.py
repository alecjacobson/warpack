"""Restarted complex Arnoldi / Rayleigh–Ritz for general sparse matrices.

Uses the pure-Warp projected Schur reference backend. Complex data are vec2d.
This path emphasizes coverage; benchmark it separately from the symmetric path.
"""

import warp as wp

from .hermitian import (
    HermitianCSR,
    cdot,
    cmul,
    complex_dots,
    complex_norm,
    complex_subtract,
    norm2,
)
from .kernels import loop_update
from .schur import cabs, schur


@wp.kernel
def complex_csr_mv(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    a: wp.array[wp.vec2d],
    x: wp.array2d[wp.vec2d],
    y: wp.array2d[wp.vec2d],
):
    j, i = wp.tid()
    val = wp.vec2d(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        val += cmul(a[e], x[j, cols[e]])
    y[j, i] = val


class ComplexCSR(HermitianCSR):
    """Complex CSR for a general square matrix, without a symmetry assumption."""

    def apply_complex(self, x, y):
        wp.launch(
            complex_csr_mv,
            x.shape,
            [self.offsets, self.columns, self.values, x, y],
            device=self.device,
        )


@wp.kernel
def random_vector(q: wp.array[wp.vec2d], seed: int):
    i = wp.tid()
    state = wp.rand_init(seed, i)
    q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))


@wp.kernel
def scale_vector(q: wp.array[wp.vec2d], norm: wp.array[wp.float64]):
    i = wp.tid()
    q[i] = q[i] / wp.sqrt(wp.max(norm[0], wp.float64(1.0e-300)))


@wp.kernel
def breakdown_vector(
    q: wp.array[wp.vec2d],
    norm: wp.array[wp.float64],
    reference: wp.array[wp.float64],
    seed: int,
):
    i = wp.tid()
    if norm[0] <= wp.float64(1.0e-26) * reference[0]:
        state = wp.rand_init(seed, i)
        q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))


@wp.kernel
def complex_gram(x: wp.array2d[wp.vec2d], y: wp.array2d[wp.vec2d], h: wp.array2d[wp.vec2d]):
    a, b, chunk = wp.tid()
    xx = wp.tile_load(x[a], shape=256, offset=chunk * 256)
    yy = wp.tile_load(y[b], shape=256, offset=chunk * 256)
    wp.tile_atomic_add(h[a], wp.tile_sum(wp.tile_map(cdot, xx, yy)), offset=b)


@wp.kernel
def sort_complex(values: wp.array[wp.vec2d], order: wp.array[wp.int32], which: int):
    n = values.shape[0]
    for i in range(n):
        order[i] = i
    for i in range(n):
        best = i
        for j in range(i + 1, n):
            a = values[order[j]]
            b = values[order[best]]
            if which == 0 and norm2(a) > norm2(b):
                best = j
            if which == 1 and norm2(a) < norm2(b):
                best = j
            if which == 2 and a[0] > b[0]:
                best = j
            if which == 3 and a[0] < b[0]:
                best = j
            if which == 4 and a[1] > b[1]:
                best = j
            if which == 5 and a[1] < b[1]:
                best = j
        t = order[i]
        order[i] = order[best]
        order[best] = t


@wp.kernel
def rotate_complex(
    q: wp.array2d[wp.vec2d],
    p: wp.array2d[wp.vec2d],
    order: wp.array[wp.int32],
    out: wp.array2d[wp.vec2d],
):
    j, i = wp.tid()
    val = wp.vec2d(0.0)
    for r in range(q.shape[0]):
        val += cmul(q[r, i], p[r, order[j]])
    out[j, i] = val


@wp.kernel
def complex_residual(
    x: wp.array2d[wp.vec2d],
    ax: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    xx = wp.tile_load(x[j], shape=256, offset=chunk * 256)
    aa = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    lam = wp.tile_full(shape=256, value=values[order[j]], dtype=wp.vec2d)
    r = aa - wp.tile_map(cmul, xx, lam)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(norm2, r)))


@wp.kernel
def complex_outputs(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    out: wp.array[wp.vec2d],
    errors: wp.array[wp.float64],
    tol: float,
    converged: wp.array[wp.int32],
):
    j = wp.tid()
    val = values[order[j]]
    err = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(val))
    out[j] = val
    errors[j] = err
    if err <= wp.float64(tol):
        wp.atomic_add(converged, 0, 1)


class GeneralEigensolver:
    """Complex nonsymmetric eigenpairs with LM/SM/LR/SR/LI/SI selection.

    SM interior targets generally need a shift-inverse operator for reliable
    convergence. No host-side eigensolve, factorization, or convergence checks.
    Inspect device `status` (0 success, 2 projected QR failure) and `converged`.
    Right eigenvectors of a general matrix need not be orthogonal.
    """

    def __init__(self, operator, k, ncv=None, which="LM", tol=1.0e-9):
        self.op = operator
        self.n = operator.complex_n
        self.device = operator.device
        self.k = k
        self.m = min(self.n, ncv or max(2 * k + 12, 32))
        self.keep = min(k + 4, self.m - 1)
        if not 1 <= k < self.m <= self.n:
            raise ValueError("require 1 <= k < ncv <= n")
        self.which = ("LM", "SM", "LR", "SR", "LI", "SI").index(which)
        self.tol = tol

        def zero(shape, dtype=wp.vec2d):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.q, self.z = zero((self.m, self.n)), zero((self.m, self.n))
        self.qrows = [self.q[i] for i in range(self.m)]
        self.views = [self.q[i : i + 1] for i in range(self.m)]
        self.zviews = [self.z[i : i + 1] for i in range(self.m)]
        self.h, self.schur_vectors, self.projected = (
            zero((self.m, self.m)),
            zero((self.m, self.m)),
            zero((self.m, self.m)),
        )
        self.ritz, self.u, self.ss = zero(self.m), zero(self.m), zero(self.m)
        self.cs = zero(self.m, wp.float64)
        self.order = zero(self.m, wp.int32)
        self.dots = zero((self.m, 1))
        self.norm = zero(1, wp.float64)
        self.reference = zero(1, wp.float64)
        self.count = zero(1, wp.int32)
        self.status = zero(1, wp.int32)
        self.converged = zero(1, wp.int32)
        self.iterations = zero(1, wp.int32)
        self.running = zero(1, wp.int32)
        self.retained = zero((self.keep, self.n))
        self.eigenvectors = zero((k, self.n))
        self.ax = zero((k, self.n))
        self.eigenvalues = zero(k)
        self.residuals = zero(k, wp.float64)
        self.norms = zero((k, 1), wp.float64)

    def length(self, j, target):
        target.zero_()
        wp.launch_tiled(
            complex_norm,
            dim=(self.n + 255) // 256,
            inputs=[self.qrows[j], target],
            block_dim=128,
            device=self.device,
        )

    def orthogonalize(self, j):
        self.count.fill_(j)
        if j:
            for _ in range(2):
                self.dots.zero_()
                wp.launch_tiled(
                    complex_dots,
                    dim=(j, (self.n + 255) // 256),
                    inputs=[self.q, self.qrows[j], self.dots],
                    block_dim=128,
                    device=self.device,
                )
                wp.launch(
                    complex_subtract,
                    self.n,
                    [self.q, self.qrows[j], self.dots, self.count],
                    device=self.device,
                )
        self.length(j, self.norm)

    def expand(self, j):
        self.op.apply_complex(self.views[j - 1], self.zviews[j - 1])
        wp.copy(self.views[j], self.zviews[j - 1])
        self.length(j, self.reference)
        self.orthogonalize(j)
        wp.launch(
            breakdown_vector,
            self.n,
            [self.qrows[j], self.norm, self.reference, 701 + j],
            device=self.device,
        )
        self.orthogonalize(j)
        wp.launch(scale_vector, self.n, [self.qrows[j], self.norm], device=self.device)

    def initialize(self, seed=42):
        self.status.zero_()
        wp.launch(random_vector, self.n, [self.qrows[0], seed], device=self.device)
        self.length(0, self.norm)
        wp.launch(scale_vector, self.n, [self.qrows[0], self.norm], device=self.device)
        for j in range(1, self.keep):
            self.expand(j)

    def step(self):
        for j in range(self.keep - 1):
            self.op.apply_complex(self.views[j], self.zviews[j])
        for j in range(self.keep, self.m):
            self.expand(j)
        self.op.apply_complex(self.views[-1], self.zviews[-1])
        self.h.zero_()
        wp.launch_tiled(
            complex_gram,
            dim=(self.m, self.m, (self.n + 255) // 256),
            inputs=[self.q, self.z, self.h],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            schur,
            1,
            [
                self.h,
                self.schur_vectors,
                self.projected,
                self.ritz,
                self.u,
                self.cs,
                self.ss,
                self.status,
                self.m * 100,
            ],
            device=self.device,
        )
        wp.launch(sort_complex, 1, [self.ritz, self.order, self.which], device=self.device)
        wp.launch(
            rotate_complex,
            self.eigenvectors.shape,
            [self.q, self.projected, self.order, self.eigenvectors],
            device=self.device,
        )
        wp.launch(
            rotate_complex,
            self.retained.shape,
            [self.q, self.projected, self.order, self.retained],
            device=self.device,
        )
        wp.copy(self.q, self.retained, count=self.keep * self.n)
        for j in range(self.keep):
            self.orthogonalize(j)
            wp.launch(scale_vector, self.n, [self.qrows[j], self.norm], device=self.device)

    def finalize(self):
        self.op.apply_complex(self.eigenvectors, self.ax)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.ritz, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            complex_outputs,
            self.k,
            [
                self.ritz,
                self.order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

    def solve(self, iterations=20):
        self.initialize()
        for _ in range(iterations):
            self.step()
        self.finalize()
        return self

    def capture(self, max_iterations=100, adaptive=None, finalizer=None):
        if adaptive is None:
            adaptive = not getattr(self.op, "requires_static_outer_capture", False)
        if adaptive and getattr(self.op, "requires_static_outer_capture", False):
            raise ValueError("This backend needs a fixed outer graph")
        self.solve(1)
        if finalizer is not None:
            finalizer()
        wp.synchronize_device(self.device)

        def body():
            self.step()
            self.finalize()
            wp.launch(
                loop_update,
                1,
                [
                    self.converged,
                    self.status,
                    self.iterations,
                    self.running,
                    self.k,
                    max_iterations,
                ],
                device=self.device,
            )

        with wp.ScopedCapture(device=self.device) as cap:
            self.initialize()
            self.iterations.zero_()
            self.running.fill_(1)
            if adaptive:
                wp.capture_while(self.running, body)
            else:
                for _ in range(max_iterations):
                    self.step()
                self.finalize()
                self.iterations.fill_(max_iterations)
            if finalizer is not None:
                finalizer()
        self.graph = cap.graph
        return self.graph
