"""Generalized SPD-mass eigenproblems and a pure-Warp inverse operator."""

import warp as wp
import warp.optim.linear as linear

from . import kernels as K
from .solver import SymmetricEigensolver


class CGInverse:
    """Pure-Warp SPD inverse via preallocated CG with device convergence checks.

    Accuracy depends on the inner solve: choose a tighter inner tolerance than
    the eigenpair tolerance and inspect the original-problem residuals.
    """

    requires_static_outer_capture = True

    def __init__(self, matrix, tol=1.0e-12, maxiter=1000):
        self.n, self.device = matrix.shape[0], matrix.device
        self.b = wp.zeros(self.n, dtype=wp.float64, device=self.device)
        self.x = wp.zeros_like(self.b)
        self.state = linear.cg(
            matrix, self.b, self.x, tol=tol, maxiter=maxiter, check_every=0, run=False
        )

    def apply(self, x, y):
        for j in range(x.shape[0]):
            y[j].zero_()
            self.state(b=x[j], x=y[j])


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


class GeneralizedEigensolver(SymmetricEigensolver):
    """Solve A x = lambda B x for symmetric A and SPD B.

    B-orthogonal inverse iteration, B-CholeskyQR2, and a projected symmetric
    eigensolve. `inverse` applies (A-sigma B)^-1. Mass need not be diagonal.
    This class targets smallest eigenvalues for SPD A, or eigenvalues near a
    provided shift with target=sigma. The supplied inverse must match that shift.
    """

    def __init__(self, operator, mass, inverse, k, ncv=None, tol=1.0e-9, which="SA", target=None):
        super().__init__(
            operator, k, ncv=ncv, which=which, iteration_operator=inverse, tol=tol, target=target
        )
        if mass.n != self.n or mass.device != self.device:
            raise ValueError("mass and stiffness must have matching shape and device")
        self.mass = mass
        self.bq = wp.empty_like(self.q)
        self.bx = wp.empty_like(self.eigenvectors)
        self.general_norms = wp.zeros((k, 3), dtype=wp.float64, device=self.device)

    def orthogonalize(self):
        for _ in range(2):
            self.mass.apply(self.q, self.bq)
            self.gram(self.q, self.bq)
            wp.launch(K.cholesky, 1, [self.g, self.l, self.status], device=self.device)
            wp.launch(K.triangular, self.n, [self.q, self.l], device=self.device)

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
