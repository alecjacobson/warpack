"""Generalized SPD-mass eigenproblems and a pure-Warp inverse operator."""

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
    """

    requires_static_outer_capture = True

    def __init__(self, matrix, tol=1.0e-12, maxiter=1000, preconditioner=None):
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
            maxiter=maxiter,
            check_every=0,
            run=False,
        )

    def reset_stats(self):
        self.stats.zero_()

    def apply(self, x, y):
        for j in range(x.shape[0]):
            y[j].zero_()
            iterations, residual_sq, tolerance_sq = self.state(b=x[j], x=y[j])
            wp.launch(
                accumulate_cg_statistics,
                1,
                [iterations, residual_sq, tolerance_sq, self.stats],
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
