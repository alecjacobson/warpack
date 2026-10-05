"""Generalized buckling and Cayley spectral transformations."""

import warp as wp

from .generalized import GeneralizedEigensolver, generalized_outputs, generalized_residual
from .kernels import identity_order


@wp.kernel
def reciprocal(x: wp.array[wp.float64], y: wp.array[wp.float64]):
    i = wp.tid()
    y[i] = wp.float64(1.0) / x[i]


@wp.kernel
def combine(
    a: wp.array2d[wp.float64],
    b: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
    sigma: wp.float64,
):
    j, i = wp.tid()
    out[j, i] = a[j, i] + sigma * b[j, i]


@wp.kernel
def cayley_order(values: wp.array[wp.float64], order: wp.array[wp.int32], sigma: wp.float64):
    n = values.shape[0]
    for i in range(n):
        order[i] = i
    for i in range(n):
        best = i
        for j in range(i + 1, n):
            a = values[order[j]]
            b = values[order[best]]
            aa = wp.abs((a + sigma) / (a - sigma))
            bb = wp.abs((b + sigma) / (b - sigma))
            if aa > bb:
                best = j
        temp = order[i]
        order[i] = order[best]
        order[best] = temp


class CayleyEigensolver(GeneralizedEigensolver):
    """A x = lambda B x, B SPD, ranked by |(lambda+sigma)/(lambda-sigma)|.

    Supply an inverse of A-sigma B. Nonzero sigma is required.
    Returned eigenvalues/residuals refer to the original generalized problem.
    """

    def __init__(self, operator, mass, inverse, k, sigma, ncv=None, tol=1.0e-9):
        if sigma == 0:
            raise ValueError("Cayley sigma must be nonzero")
        self.sigma = float(sigma)
        super().__init__(operator, mass, inverse, k, ncv=ncv, tol=tol)
        self.rhs = wp.empty_like(self.q)

    def step(self):
        self.op.apply(self.q, self.z)
        self.mass.apply(self.q, self.bq)
        wp.launch(
            combine, self.q.shape, [self.z, self.bq, self.rhs, self.sigma], device=self.device
        )
        self.iter_op.apply(self.rhs, self.z)
        wp.copy(self.q, self.z)
        self.orthogonalize()

    def diagonalize(self):
        self._diagonalize_projected()
        wp.launch(cayley_order, 1, [self.ritz, self.order, self.sigma], device=self.device)


class BucklingEigensolver:
    """A x = lambda B x, A SPD and B symmetric (possibly indefinite).

    Supply an inverse of A-sigma B. Selects largest transformed magnitudes
    |lambda/(lambda-sigma)|, as in ARPACK buckling mode. Eigenvectors are
    A-orthonormal. This selection is not identical to nearest physical lambda.
    """

    def __init__(self, operator, mass, inverse, k, sigma, ncv=None, tol=1.0e-9):
        if sigma == 0:
            raise ValueError("Buckling sigma must be nonzero")
        self.a, self.b, self.k, self.tol = operator, mass, k, tol
        self.device = operator.device
        self.core = GeneralizedEigensolver(
            mass, operator, inverse, k, ncv=ncv, target=1.0 / sigma, tol=tol * 0.1
        )
        self.eigenvectors = self.core.eigenvectors
        self.eigenvalues = wp.empty(k, dtype=wp.float64, device=self.device)
        self.values = wp.empty_like(self.eigenvalues)
        self.residuals = wp.empty_like(self.eigenvalues)
        self.norms = wp.zeros((k, 3), dtype=wp.float64, device=self.device)
        self.ax = wp.empty_like(self.eigenvectors)
        self.bx = wp.empty_like(self.eigenvectors)
        self.order = wp.empty(k, dtype=wp.int32, device=self.device)
        self.converged = wp.zeros(1, dtype=wp.int32, device=self.device)

    def finalize(self):
        wp.launch(reciprocal, self.k, [self.core.eigenvalues, self.values], device=self.device)
        wp.launch(identity_order, self.k, [self.order], device=self.device)
        self.a.apply(self.eigenvectors, self.ax)
        self.b.apply(self.eigenvectors, self.bx)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (self.a.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.bx, self.values, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            generalized_outputs,
            self.k,
            [
                self.values,
                self.order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

    def solve(self, iterations=50):
        self.core.solve(iterations)
        self.finalize()
        return self

    def capture(self, iterations=50):
        self.graph = self.core.capture(iterations, adaptive=False, finalizer=self.finalize)
        return self.graph
