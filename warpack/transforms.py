"""Generalized buckling and Cayley spectral transformations."""

import warp as wp

from .generalized import GeneralizedEigensolver, generalized_outputs, generalized_residual
from .generalized_krylov import GeneralizedKrylovSchur
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

    Args:
        operator: Real symmetric operator exposing n, device and apply(x, y) on FP64 row arrays.
        mass: SPD mass/metric operator on the same device and dimension as operator.
        inverse: Caller-owned block inverse of A-sigma*M accepting ncv RHSs.
        k: Number of requested eigenpairs (positive integer).
        sigma: Nonzero real shift matching the supplied inverse.
        ncv: Block width; None uses min(n, max(2*k, k+8)); require k <= ncv <= n.
        tol: Positive tolerance for ||A*x-lambda*M*x|| / max(1, ||A*x||+abs(lambda)*||M*x||);
            use B for the buckling metric. Residual convergence alone does not certify spectrum
            completeness.


    Attributes:
        eigenvalues: FP64 device array (k,) of original-problem eigenvalues.
        eigenvectors: FP64 device array (k, n); one eigenvector per row.
        residuals: FP64 device array (k,) of normalized eigenpair residuals.
        converged: int32 device array (1,); number of residual-passing requested pairs.
        status: int32 device array (1,); zero means no recorded numerical failure, not
            convergence.
        iterations: int32 device array (1,); executed outer iterations/restart cycles.
    """

    def __init__(self, operator, mass, inverse, k, sigma, ncv=None, tol=1.0e-9):
        if sigma == 0:
            raise ValueError("Cayley sigma must be nonzero")
        self.sigma = float(sigma)
        super().__init__(operator, mass, inverse, k, ncv=ncv, tol=tol)
        self.rhs = wp.empty_like(self.q)

    def step(self):
        """Advance one outer iteration or Krylov restart cycle using existing device state.

        Returns:
            None. Results are written to the documented device arrays.
        """
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


@wp.kernel
def scale_inverse_result(x: wp.array2d[wp.float64], scale: wp.float64):
    i, j = wp.tid()
    x[i, j] *= scale


class _ScaledInverse:
    def __init__(self, inverse, scale):
        self.inverse, self.scale = inverse, float(scale)
        self.n, self.device = inverse.n, inverse.device
        self.width = getattr(inverse, "width", 1)
        self.requires_static_outer_capture = getattr(
            inverse, "requires_static_outer_capture", False
        )

    def apply(self, x, y):
        self.inverse.apply(x, y)
        wp.launch(scale_inverse_result, y.shape, [y, self.scale], device=self.device)


class BucklingEigensolver:
    """A x = lambda B x, A SPD and B symmetric (possibly indefinite).

    Supply an inverse of A-sigma B. Selects largest transformed magnitudes
    |lambda/(lambda-sigma)|, as in ARPACK buckling mode. Eigenvectors are
    A-orthonormal. This selection is not identical to nearest physical lambda.
    ``method="krylov"`` uses mass-orthogonal restarted Krylov iteration and
    requires a one-RHS inverse. The default block method uses ncv RHSs.

    Args:
        operator: Real SPD operator A.
        mass: Real symmetric operator B, possibly indefinite; same dimension/device as A.
        inverse: Caller-owned inverse of A-sigma*B. One RHS for method=krylov; ncv RHSs for
            method=block.
        k: Number of requested eigenpairs (positive integer).
        sigma: Nonzero real buckling shift matching the supplied inverse.
        ncv: Iteration subspace width; default min(n, max(2*k, k+8)). Krylov requires ncv > k.
        tol: Positive tolerance for ||A*x-lambda*M*x|| / max(1, ||A*x||+abs(lambda)*||M*x||);
            use B for the buckling metric. Residual convergence alone does not certify spectrum
            completeness.
        method: block (default) or krylov; selects the transformed outer iteration.

    Attributes:
        eigenvalues: FP64 device array (k,) of original-problem eigenvalues.
        eigenvectors: FP64 device array (k, n); one eigenvector per row.
        residuals: FP64 device array (k,) of normalized eigenpair residuals.
        converged: int32 device array (1,); number of residual-passing requested pairs.
        core: Underlying solver; inspect core.status and core.iterations. There is no top-level
            status attribute.
    """

    def __init__(self, operator, mass, inverse, k, sigma, ncv=None, tol=1.0e-9, method="block"):
        if sigma == 0:
            raise ValueError("Buckling sigma must be nonzero")
        if method not in ("block", "krylov"):
            raise ValueError("method must be block or krylov")
        self.a, self.b, self.k, self.tol = operator, mass, k, tol
        self.device, self.method = operator.device, method
        if method == "krylov":
            # Reciprocal pencil B x = mu A x has SPD metric A. Its nearest
            # target mu=1/sigma selects largest |lambda/(lambda-sigma)|.
            # (B - A/sigma)^-1 = -sigma * (A - sigma B)^-1.
            self.core = GeneralizedKrylovSchur(
                mass,
                operator,
                _ScaledInverse(inverse, -sigma),
                k,
                ncv=ncv,
                target=1.0 / sigma,
                tol=tol * 0.1,
            )
        else:
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
        """Update the documented device outputs from the current iteration state.

        Returns:
            None. Results are written to the documented device arrays.
        """
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
        """Initialize and execute a fixed iteration budget, updating device outputs.

        Args:
            iterations: Fixed number of outer iterations/restart cycles, not inner sparse-solver
                iterations.

        Returns:
            This solver instance. This is not a host tuple of eigenvalues and eigenvectors;
            outputs remain device arrays.
        """
        self.core.solve(iterations)
        self.finalize()
        return self

    def capture(self, iterations=50):
        """Prepare and record a reusable CUDA solve graph outside any active capture.

        Args:
            iterations: Fixed number of outer iterations/restart cycles, not inner sparse-solver
                iterations.

        Returns:
            A warp.Graph. Launch it with wp.capture_launch(graph); capture itself does not
            produce a completed result.
        """
        self.graph = self.core.capture(iterations, adaptive=False, finalizer=self.finalize)
        return self.graph
