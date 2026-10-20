"""Restarted complex Arnoldi / Rayleigh–Ritz for general sparse matrices.

Uses the pure-Warp projected Schur reference backend. Complex data are vec2d.
This path emphasizes coverage; benchmark it separately from the symmetric path.
"""

from operator import index as integer_index

import warp as wp

from .general_checkpoint import GeneralCheckpoint
from .hermitian import (
    HermitianCSR,
    cdot,
    cmul,
    complex_dots,
    complex_norm,
    complex_subtract,
    norm2,
    start_scale,
)
from .kernels import loop_update
from .schur import cabs, schur
from .schur_fused import FusedSchur, fused_supported
from .schur_tiled import TiledSchur


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
def copy_scaled_start(
    source: wp.array[wp.vec2d],
    target: wp.array[wp.vec2d],
    scale: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
):
    i = wp.tid()
    value = wp.vec2d(0.0)
    if scale[0] > wp.float64(0.0) and invalid[0] == 0:
        value = source[i] / scale[0]
    elif i == 0:
        # Safe work vector for an invalid start. The device status rejects
        # outputs; keeping arithmetic finite also permits changed-start replay.
        value = wp.vec2d(1.0, 0.0)
    target[i] = value


@wp.kernel
def check_start(
    scale: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
    status: wp.array[wp.int32],
    converged: wp.array[wp.int32],
):
    if scale[0] == wp.float64(0.0) or invalid[0] != 0:
        wp.atomic_max(status, 0, 1)
        converged[0] = 0


@wp.kernel
def scale_vector(q: wp.array[wp.vec2d], norm: wp.array[wp.float64]):
    i = wp.tid()
    q[i] = q[i] / wp.sqrt(wp.max(norm[0], wp.float64(1.0e-300)))


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


@wp.kernel
def select_restart_residual(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    index: wp.array[wp.int32],
):
    # Search selection is not a stopping test. In shift-invert, a small
    # transformed residual can still leave the original equation unfinished.
    best = wp.float64(0.0)
    index[0] = -1
    for j in range(norms.shape[0]):
        score = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
        if wp.isfinite(score) and score > best:
            best = score
            index[0] = j


@wp.kernel
def save_restart_residual(
    x: wp.array2d[wp.vec2d],
    ax: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    index: wp.array[wp.int32],
    out: wp.array[wp.vec2d],
):
    i = wp.tid()
    j = index[0]
    if j >= 0:
        out[i] = (ax[j, i] - cmul(values[order[j]], x[j, i])) / wp.sqrt(norms[j, 0])


@wp.kernel
def restart_direction(
    q: wp.array[wp.vec2d],
    norm: wp.array[wp.float64],
    reference: wp.array[wp.float64],
    candidate: wp.array[wp.vec2d],
    index: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    used: wp.array[wp.int32],
    use_candidate: int,
    seed: int,
):
    i = wp.tid()
    if i == 0:
        used[0] = 0
    if norm[0] <= wp.float64(1.0e-26) * reference[0] or (use_candidate == 2 and index[0] >= 0):
        if use_candidate != 0 and index[0] >= 0:
            q[i] = candidate[i]
            if i == 0:
                recoveries[0] += 1
                used[0] = 1
        else:
            state = wp.rand_init(seed, i)
            q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))


@wp.kernel
def complete_restart_direction(
    q: wp.array[wp.vec2d],
    norm: wp.array[wp.float64],
    used: wp.array[wp.int32],
    fallbacks: wp.array[wp.int32],
    seed: int,
):
    i = wp.tid()
    # The candidate was normalized before projection. If it is already in the
    # growing basis, retain random completion rather than install a zero row.
    if used[0] != 0 and (norm[0] <= wp.float64(1.0e-26) or not wp.isfinite(norm[0])):
        state = wp.rand_init(seed, i)
        q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))
        if i == 0:
            fallbacks[0] += 1


class GeneralEigensolver:
    """Complex nonsymmetric eigenpairs with LM/SM/LR/SR/LI/SI selection.

    SM interior targets generally need a shift-inverse operator for reliable
    convergence. No host-side eigensolve, factorization, or convergence checks.
    Inspect device `status` (0 success, 1 invalid start, 2 projected QR failure)
    and `converged`. Optional ``initial_vector`` is a finite nonzero ``wp.vec2d``
    array of shape ``(n,)`` on the operator device. Capture rereads its current
    contents at every replay; safe scaling and normalization run on device.
    Zero/nonfinite starts set a failure status and zero the convergence count.
    Check status before using outputs. A start does not certify target completeness.
    CUDA projections of size 16–64 use a cooperative QR loop when conditional
    graphs are supported; other sizes/devices retain the serial reference.
    Fixed budgets preserve the best complete converged set observed during
    the solve, while continuing exploration. ``preserve_converged=False`` opts
    out. Scratch is allocated before the first fixed solve/capture and reset
    on every replay. Adaptive graphs do not use checkpoints. Preservation is
    not a certificate of globally requested-spectrum completeness.
    Right eigenvectors of a general matrix need not be orthogonal.
    Restarts expand from the largest remaining requested Ritz residual, using
    cached operator images. Device ``residual_restarts`` and ``residual_fallbacks``
    count candidate uses and random completion of projected-out candidates.
    Fresh original-operator products still determine convergence.
    Solve and capture budgets must be positive integers; zero cycles cannot
    produce initialized Ritz vectors and are rejected before warmup.
    """

    def __init__(
        self,
        operator,
        k,
        ncv=None,
        which="LM",
        tol=1.0e-9,
        initial_vector=None,
        preserve_converged=True,
    ):
        self.op = operator
        self.n = operator.complex_n
        self.device = operator.device
        if initial_vector is not None:
            if (
                not isinstance(initial_vector, wp.array)
                or initial_vector.dtype != wp.vec2d
                or initial_vector.shape != (self.n,)
            ):
                raise ValueError("initial_vector must have shape (n,) and dtype wp.vec2d")
            if initial_vector.device != self.device:
                raise ValueError("initial_vector must be on the operator device")
        self.initial_vector = initial_vector
        self.preserve_converged = bool(preserve_converged)
        self._checkpoint = None
        self._checkpoint_enabled = False
        self._checkpoint_validator = None
        self.k = k
        self.m = min(self.n, ncv or max(2 * k + 12, 32))
        self.keep = min(k + 4, self.m - 1)
        if not 1 <= k < self.m <= self.n:
            raise ValueError("require 1 <= k < ncv <= n")
        self.which = ("LM", "SM", "LR", "SR", "LI", "SI").index(which)
        self.tol = tol

        def zero(shape, dtype=wp.vec2d):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        if initial_vector is not None:
            self.start_scale = zero(1, wp.float64)
            self.start_invalid = zero(1, wp.int32)
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
        self.restart_residual = zero(self.n)
        self.restart_index = wp.full(1, -1, dtype=wp.int32, device=self.device)
        self.residual_restarts = zero(1, wp.int32)
        self.recovery_used = zero(1, wp.int32)
        self.residual_fallbacks = zero(1, wp.int32)
        self.projected_solver = None
        if self.device.is_cuda and 16 <= self.m <= 64 and wp.is_conditional_graph_supported():
            projection = FusedSchur if fused_supported(self.device, self.m) else TiledSchur
            self.projected_solver = projection(
                self.h,
                self.schur_vectors,
                self.projected,
                self.ritz,
                self.u,
                self.cs,
                self.ss,
                self.status,
            )

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
            restart_direction,
            self.n,
            [
                self.qrows[j],
                self.norm,
                self.reference,
                self.restart_residual,
                self.restart_index,
                self.residual_restarts,
                self.recovery_used,
                2 if j == self.keep else int(j > self.keep),
                701 + j,
            ],
            device=self.device,
        )
        self.orthogonalize(j)
        if j >= self.keep:
            wp.launch(
                complete_restart_direction,
                self.n,
                [self.qrows[j], self.norm, self.recovery_used, self.residual_fallbacks, 701 + j],
                device=self.device,
            )
            self.orthogonalize(j)
        wp.launch(scale_vector, self.n, [self.qrows[j], self.norm], device=self.device)

    def initialize(self, seed=42):
        if self._checkpoint is not None:
            self._checkpoint.reset()
        self.restart_index.fill_(-1)
        self.residual_restarts.zero_()
        self.residual_fallbacks.zero_()
        self.status.zero_()
        if self.initial_vector is None:
            wp.launch(random_vector, self.n, [self.qrows[0], seed], device=self.device)
        else:
            self.start_scale.zero_()
            self.start_invalid.zero_()
            wp.launch(
                start_scale,
                self.n,
                [self.initial_vector, self.start_scale, self.start_invalid],
                device=self.device,
            )
            wp.launch(
                copy_scaled_start,
                self.n,
                [self.initial_vector, self.qrows[0], self.start_scale, self.start_invalid],
                device=self.device,
            )
        self.length(0, self.norm)
        wp.launch(scale_vector, self.n, [self.qrows[0], self.norm], device=self.device)
        for j in range(1, self.keep):
            self.expand(j)
        self._check_start(self.converged)

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
        self.diagonalize()
        self.select()
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

        self.prepare_restart()

    def select(self):
        wp.launch(sort_complex, 1, [self.ritz, self.order, self.which], device=self.device)

    def diagonalize(self):
        if self.projected_solver is not None:
            self.projected_solver.run()
            return
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

    def prepare_restart(self):
        # z and projected still describe the pre-restart basis. Their rotation
        # supplies a search direction without another operator/inverse call.
        wp.launch(
            rotate_complex,
            self.ax.shape,
            [self.z, self.projected, self.order, self.ax],
            device=self.device,
        )
        self.norms.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.ritz, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            select_restart_residual,
            1,
            [self.ritz, self.order, self.norms, self.restart_index],
            device=self.device,
        )
        wp.launch(
            save_restart_residual,
            self.n,
            [
                self.eigenvectors,
                self.ax,
                self.ritz,
                self.order,
                self.norms,
                self.restart_index,
                self.restart_residual,
            ],
            device=self.device,
        )

        if self._checkpoint_enabled:
            self._checkpoint.consider(self)

    def _enable_checkpoint(self, enabled):
        self._checkpoint_enabled = bool(enabled and self.preserve_converged)
        if self._checkpoint_enabled and self._checkpoint is None:
            self._checkpoint = GeneralCheckpoint(self)

    def finalize(self):
        values, order = self.ritz, self.order
        if self._checkpoint_enabled:
            values, order = self._checkpoint.restore(self)
        self.op.apply_complex(self.eigenvectors, self.ax)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, values, order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            complex_outputs,
            self.k,
            [
                values,
                order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

        self._check_start(self.converged)

    def _check_start(self, converged):
        if self.initial_vector is not None:
            wp.launch(
                check_start,
                1,
                [self.start_scale, self.start_invalid, self.status, converged],
                device=self.device,
            )

    def solve(self, iterations=20):
        iterations = integer_index(iterations)
        if iterations < 1:
            raise ValueError("General iteration needs at least one restart cycle")
        self._enable_checkpoint(True)
        self.initialize()
        self.iterations.zero_()
        for _ in range(iterations):
            self.step()
        self.finalize()
        self.iterations.fill_(iterations)
        return self

    def capture(self, max_iterations=100, adaptive=None, finalizer=None):
        max_iterations = integer_index(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if adaptive is None:
            adaptive = not getattr(self.op, "requires_static_outer_capture", False)
        if adaptive and getattr(self.op, "requires_static_outer_capture", False):
            raise ValueError("This backend needs a fixed outer graph")
        self._enable_checkpoint(not adaptive)
        self.initialize()
        self.step()
        self.finalize()
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
