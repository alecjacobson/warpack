"""Generalized SPD-mass eigenproblems and a pure-Warp inverse operator."""

import math
from operator import index

import warp as wp
import warp.optim.linear as linear

from . import kernels as K
from .solver import SymmetricEigensolver


@wp.kernel
def accumulate_cg_statistics(
    iterations: wp.array[wp.int32],
    residual_sq: wp.array[wp.float64],
    tolerance_sq: wp.array[wp.float64],
    stats: wp.array[wp.int32],
):
    wp.atomic_add(stats, 0, iterations[0])
    wp.atomic_max(stats, 1, iterations[0])
    if residual_sq[0] > tolerance_sq[0] or not wp.isfinite(residual_sq[0]):
        wp.atomic_add(stats, 2, 1)
    wp.atomic_add(stats, 3, 1)


@wp.kernel
def accumulate_refinement_stage(
    iterations: wp.array[wp.int32],
    residual_sq: wp.array[wp.float64],
    tolerance_sq: wp.array[wp.float64],
    stage: wp.array[wp.int32],
    native: wp.array[wp.int32],
):
    failed = int(residual_sq[0] > tolerance_sq[0] or not wp.isfinite(residual_sq[0]))
    stage[0] += iterations[0]
    stage[1] = wp.max(stage[1], iterations[0])
    stage[2] += failed
    stage[3] += 1
    native[0] += iterations[0]
    native[1] = wp.max(native[1], iterations[0])
    native[2] += failed
    native[3] += 1


@wp.kernel
def finish_refinement_statistics(stage: wp.array[wp.int32], stats: wp.array[wp.int32]):
    stats[0] += stage[0]
    stats[1] = wp.max(stats[1], stage[0])
    stats[2] += int(stage[2] > 0)
    stats[3] += 1


def _cg_preallocates_loop_control(state):
    # Warp 1.15 allocates conditional-loop storage inside each solve, which
    # CUDA forbids inside an outer conditional body. The tested 1.18 state
    # preallocates it. No public capability flag exists; fail conservatively
    # if a future upstream layout no longer exposes this storage.
    return getattr(state, "_cur_iter_and_condition", None) is not None


class CGInverse:
    """Pure-Warp SPD inverse with optional upstream Warp preconditioning.

    Pass a Warp ``preconditioner`` name, such as ``"block_jacobi_sequential"``
    on versions supporting it, or a prebuilt Warp linear operator (e.g. FSAI).
    Unsupported names raise Warp's own ValueError.
    Recreate the inverse when matrix values change to refresh this preconditioner.
    ``stats`` holds device counters (total iterations, maximum iterations per RHS,
    unconverged RHSs, number of RHSs); reset explicitly before a measured replay.
    Inner stopping uses device convergence checks. Always verify the original
    eigenproblem residual: a converged recursive CG residual can be optimistic.

    ``verify_residuals=True`` additionally recomputes each original linear
    residual on device. ``fresh_stats`` uses the same four counters as ``stats``
    but checks that fresh residual against upstream's effective tolerance;
    ``residual_maxima`` holds maximum fresh absolute/relative residual norms.
    The upstream solve and recursive ``stats`` are unchanged. Both additional
    arrays are ``None`` when this optional audit is disabled.

    Exact in-place application is supported. Nonidentical input/output views
    with overlapping address ranges are rejected.

    ``atol=None`` preserves upstream Warp's default: ``tol`` is both the
    relative and absolute residual tolerance. Pass ``atol=0.0`` for relative
    stopping on small right-hand sides, such as eigenvector residuals used
    for preconditioning. Warp still applies its dtype-dependent safety floor.

    Setting ``refinement_steps>0`` adds a fixed number of fresh-residual
    corrections. Each uses another upstream CG state with the same
    preconditioner, relative ``refinement_tol`` (default 0.01), zero absolute
    tolerance and the same maximum iteration budget. No correction is enabled
    by default. Corrections need not attain an arbitrarily strict residual.
    With corrections, ``stats`` counts original RHS calls, total initial plus
    correction iterations, and a failed RHS if any constituent CG state fails.
    ``initial_stats`` and ``refinement_stats`` separately expose the native
    state totals; both are None when refinement is disabled. Optional fresh
    auditing checks the final solution against the original RHS and the
    initial state's effective tolerance, not the correction RHS tolerance.

    ``adaptive_capture=True`` opts into nested device-controlled outer graphs
    when the upstream CG state preallocates loop control (tested on Warp 1.18).
    Older states, including stock Warp 1.15, reject the opt-in. The default
    preserves fixed outer capture for compatibility.
    """

    requires_static_outer_capture = True

    def __init__(
        self,
        matrix,
        tol=1.0e-12,
        maxiter=1000,
        preconditioner=None,
        atol=None,
        adaptive_capture=False,
        verify_residuals=False,
        refinement_steps=0,
        refinement_tol=0.01,
    ):
        if isinstance(refinement_steps, bool) or index(refinement_steps) < 0:
            raise ValueError("refinement_steps must be a nonnegative integer")
        if not math.isfinite(refinement_tol) or not 0 < refinement_tol < 1:
            raise ValueError("refinement_tol must be finite and between zero and one")
        self.refinement_steps = index(refinement_steps)
        self.refinement_tol = refinement_tol
        if atol is not None and (not math.isfinite(atol) or atol < 0):
            raise ValueError("atol must be finite and nonnegative, or None")
        self.n, self.device = matrix.shape[0], matrix.device
        self.b = wp.zeros(self.n, dtype=wp.float64, device=self.device)
        self.x = wp.zeros_like(self.b)
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.preconditioner = None
        if preconditioner is not None:
            # Use Warp's upstream implementation; no local factorization kernels.
            self.preconditioner = (
                linear.preconditioner(matrix, ptype=preconditioner)
                if isinstance(preconditioner, str)
                else preconditioner
            )
        self.state = linear.cg(
            matrix,
            self.b,
            self.x,
            M=self.preconditioner,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            check_every=0,
            run=False,
        )

        if adaptive_capture and not _cg_preallocates_loop_control(self.state):
            raise ValueError("adaptive_capture requires upstream CG with preallocated loop control")
        self.requires_static_outer_capture = not adaptive_capture
        self.verify_residuals = bool(verify_residuals)
        self.fresh_stats = None
        self.residual_maxima = None
        self.initial_stats = None
        self.refinement_stats = None
        if self.verify_residuals or self.refinement_steps:
            self._audit_operator = linear.aslinearoperator(matrix)
            self._audit_ax = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        if self.verify_residuals:
            # Shared with GMRES; imported at setup to avoid the module cycle
            # through accumulate_cg_statistics. No Python callback on replay.
            from .iterative import residual_maxima, residual_norms

            self._residual_norms_kernel = residual_norms
            self._residual_maxima_kernel = residual_maxima
            self._audit_norms = wp.zeros(2, dtype=wp.float64, device=self.device)
            self.fresh_stats = wp.zeros(4, dtype=wp.int32, device=self.device)
            self.residual_maxima = wp.zeros(2, dtype=wp.float64, device=self.device)
        if self.refinement_steps:
            self._correction = wp.empty_like(self._audit_ax)
            self._refinement_state = linear.cg(
                matrix,
                self._audit_ax[0],
                self._correction[0],
                M=self.preconditioner,
                tol=refinement_tol,
                atol=0.0,
                maxiter=maxiter,
                check_every=0,
                run=False,
            )
            if adaptive_capture and not _cg_preallocates_loop_control(self._refinement_state):
                raise ValueError("adaptive_capture requires preallocated refinement loop control")
            self._refinement_stage = wp.zeros(4, dtype=wp.int32, device=self.device)
            self._refinement_iterations = self._refinement_stage[:1]
            self.initial_stats = wp.zeros_like(self._refinement_stage)
            self.refinement_stats = wp.zeros_like(self._refinement_stage)

    def reset_stats(self):
        self.stats.zero_()
        if self.refinement_steps:
            self.initial_stats.zero_()
            self.refinement_stats.zero_()
        if self.verify_residuals:
            self.fresh_stats.zero_()
            self.residual_maxima.zero_()

    def apply(self, x, y):
        if (
            x.ndim != 2
            or x.shape != y.shape
            or x.shape[1] != self.n
            or x.dtype != wp.float64
            or y.dtype != wp.float64
            or x.device != self.device
            or y.device != self.device
        ):
            raise ValueError("expected matching FP64 (width,n) device rows")
        if x.shape[0] == 0:
            return
        alias = x.ptr == y.ptr and x.strides == y.strides

        def span(array):
            # Address metadata only, including strided views. Never read a
            # device scalar to decide the capture/replay path.
            offsets = [(n - 1) * stride for n, stride in zip(array.shape, array.strides)]
            return (
                array.ptr + sum(min(0, offset) for offset in offsets),
                array.ptr + sum(max(0, offset) for offset in offsets) + 8,
            )

        if not alias:
            xlo, xhi = span(x)
            ylo, yhi = span(y)
            if max(xlo, ylo) < min(xhi, yhi):
                raise ValueError("partially overlapping CG input/output views are unsupported")
        snapshot = self.b
        if alias or self.verify_residuals or self.refinement_steps:
            # Also allow callers to use this inverse's exposed b/x workspaces.
            # The snapshot must not be the output that is about to be cleared.
            ylo, yhi = span(y)
            blo, bhi = span(self.b)
            if max(ylo, blo) < min(yhi, bhi):
                snapshot = self.x
        for j in range(x.shape[0]):
            rhs = x[j]
            if alias or self.verify_residuals or self.refinement_steps:
                wp.copy(snapshot, rhs)
                rhs = snapshot
            y[j].zero_()
            iterations, residual_sq, tolerance_sq = self.state(b=rhs, x=y[j])
            if self.refinement_steps:
                self._refinement_stage.zero_()
                wp.launch(
                    accumulate_refinement_stage,
                    1,
                    [
                        iterations,
                        residual_sq,
                        tolerance_sq,
                        self._refinement_stage,
                        self.initial_stats,
                    ],
                    device=self.device,
                )
                for _ in range(self.refinement_steps):
                    self._audit_operator.matvec(
                        y[j], self._audit_ax[0], self._audit_ax[0], 1.0, 0.0
                    )
                    wp.launch(
                        K.linear_residual,
                        (1, self.n),
                        [rhs.reshape((1, self.n)), self._audit_ax],
                        device=self.device,
                    )
                    self._correction.zero_()
                    correction_iterations, correction_residual, correction_tolerance = (
                        self._refinement_state(b=self._audit_ax[0], x=self._correction[0])
                    )
                    wp.launch(
                        accumulate_refinement_stage,
                        1,
                        [
                            correction_iterations,
                            correction_residual,
                            correction_tolerance,
                            self._refinement_stage,
                            self.refinement_stats,
                        ],
                        device=self.device,
                    )
                    wp.launch(
                        K.add_correction,
                        (1, self.n),
                        [y[j : j + 1], self._correction],
                        device=self.device,
                    )
                wp.launch(
                    finish_refinement_statistics,
                    1,
                    [self._refinement_stage, self.stats],
                    device=self.device,
                )
                iterations = self._refinement_iterations
            else:
                wp.launch(
                    accumulate_cg_statistics,
                    1,
                    [iterations, residual_sq, tolerance_sq, self.stats],
                    device=self.device,
                )
            if self.verify_residuals:
                self._audit_operator.matvec(y[j], self._audit_ax[0], self._audit_ax[0], 1.0, 0.0)
                self._audit_norms.zero_()
                wp.launch_tiled(
                    self._residual_norms_kernel,
                    dim=(self.n + 255) // 256,
                    inputs=[self._audit_ax[0], rhs, self._audit_norms],
                    block_dim=64,
                    device=self.device,
                )
                wp.launch(
                    accumulate_cg_statistics,
                    1,
                    [iterations, self._audit_norms, tolerance_sq, self.fresh_stats],
                    device=self.device,
                )
                wp.launch(
                    self._residual_maxima_kernel,
                    1,
                    [self._audit_norms, self.residual_maxima],
                    device=self.device,
                )


@wp.kernel
def generalized_residual(
    x: wp.array2d[wp.float64],
    ax: wp.array2d[wp.float64],
    bx: wp.array2d[wp.float64],
    vals: wp.array[wp.float64],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    a = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(bx[j], shape=256, offset=chunk * 256)
    r = a - b * vals[order[j]]
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, r, r)), offset=0)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, a, a)), offset=1)
    wp.tile_atomic_add(norms[j], wp.tile_sum(wp.tile_map(wp.mul, b, b)), offset=2)


@wp.kernel
def generalized_outputs(
    vals: wp.array[wp.float64],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    out: wp.array[wp.float64],
    errors: wp.array[wp.float64],
    tol: float,
    converged: wp.array[wp.int32],
):
    j = wp.tid()
    lam = vals[order[j]]
    denominator = wp.max(wp.float64(1.0), wp.sqrt(norms[j, 1]) + wp.abs(lam) * wp.sqrt(norms[j, 2]))
    err = wp.sqrt(norms[j, 0]) / denominator
    out[j] = lam
    errors[j] = err
    if err <= wp.float64(tol):
        wp.atomic_add(converged, 0, 1)


@wp.kernel
def metric_cholesky(
    g: wp.array2d[wp.float64],
    factor: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    repaired: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    allow_shift: bool,
):
    # A finite inverse image can have a Gram matrix that is indefinite in FP64.
    # Retry before applying the damaged factor to the basis. The trace-scaled
    # shift is a numerical safeguard, not a change to the eigenproblem.
    shift = wp.float64(0.0)
    for attempt in range(2):
        bad = int(0)
        for i in range(g.shape[0]):
            for j in range(i + 1):
                v = g[i, j]
                if i == j:
                    v += shift
                for k in range(j):
                    v -= factor[i, k] * factor[j, k]
                if i == j:
                    if not wp.isfinite(v) or v <= wp.float64(0.0):
                        bad = 1
                        break
                    if allow_shift and attempt == 0 and v <= wp.float64(1.0e-14) * g[i, i]:
                        bad = 1
                        break
                    factor[i, j] = wp.sqrt(v)
                else:
                    factor[i, j] = v / factor[j, j]
            if bad != 0:
                break
        if bad == 0:
            break
        if not allow_shift or attempt == 1:
            status[0] = 1
            break
        trace = wp.float64(0.0)
        for i in range(g.shape[0]):
            trace += g[i, i]
        shift = wp.float64(1.0e-12) * trace
        repaired[0] = 1
        recoveries[0] += 1


class GeneralizedEigensolver(SymmetricEigensolver):
    """Solve A x = lambda B x for symmetric A and SPD B.

    B-orthogonal inverse iteration, stabilized B-CholeskyQR2, and a projected symmetric
    eigensolve. `inverse` applies (A-sigma B)^-1. Mass need not be diagonal.
    This class targets smallest eigenvalues for SPD A, or eigenvalues near a
    provided shift with target=sigma. The supplied inverse must match that shift.
    Ill-conditioned Gram factors are retried with a small diagonal shift, then
    two unshifted metric QR passes restore orthogonality. Recovery decisions and
    ``orthogonalization_recoveries`` stay on device. This safeguard does not
    guarantee recovery of rank-deficient blocks; inspect ``status`` as well as
    physical residuals and the requested spectrum.
    """

    def __init__(
        self,
        operator,
        mass,
        inverse,
        k,
        ncv=None,
        tol=1.0e-9,
        which="SA",
        target=None,
        initial_vectors=None,
    ):
        super().__init__(
            operator,
            k,
            ncv=ncv,
            which=which,
            iteration_operator=inverse,
            tol=tol,
            target=target,
            initial_vectors=initial_vectors,
        )
        if mass.n != self.n or mass.device != self.device:
            raise ValueError("mass and stiffness must have matching shape and device")
        self.mass = mass
        self.bq = wp.empty_like(self.q)
        self.bx = wp.empty_like(self.eigenvectors)
        self.general_norms = wp.zeros((k, 3), dtype=wp.float64, device=self.device)
        self._orth_repaired = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.orthogonalization_recoveries = wp.zeros(1, dtype=wp.int32, device=self.device)

    def initialize(self, seed=42):
        self.orthogonalization_recoveries.zero_()
        super().initialize(seed)

    def _metric_qr(self, allow_shift):
        self.mass.apply(self.q, self.bq)
        self.gram(self.q, self.bq)
        wp.launch(
            metric_cholesky,
            1,
            [
                self.g,
                self.l,
                self.status,
                self._orth_repaired,
                self.orthogonalization_recoveries,
                allow_shift,
            ],
            device=self.device,
        )
        wp.launch(K.triangular, self.n, [self.q, self.l], device=self.device)

    def orthogonalize(self):
        self._orth_repaired.zero_()
        for _ in range(2):
            self._metric_qr(True)

        def finish_qr2():
            for _ in range(2):
                self._metric_qr(False)

        if wp.get_stream(self.device).is_capturing:
            wp.capture_if(self._orth_repaired, finish_qr2)
        else:
            # Eager execution does not read the numerical decision on the host.
            finish_qr2()

    def step(self):
        self.mass.apply(self.q, self.bq)
        self.iter_op.apply(self.bq, self.z)
        wp.copy(self.q, self.z)
        self.orthogonalize()

    def finalize(self):
        self.op.apply(self.q, self.z)
        self.gram(self.q, self.z)
        self.diagonalize()
        self.rotate(self.q, self.eigenvectors)
        self.op.apply(self.eigenvectors, self.ax)
        self.mass.apply(self.eigenvectors, self.bx)
        self.general_norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[
                self.eigenvectors,
                self.ax,
                self.bx,
                self.ritz,
                self.order,
                self.general_norms,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            generalized_outputs,
            self.k,
            [
                self.ritz,
                self.order,
                self.general_norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )
