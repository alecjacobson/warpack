"""Exact reciprocal SPD-metric adapter; not a raw singular-mass core API."""

import numpy as np
import warp as wp
import warp.optim.linear as linear

from benchmarks.upstream.adapters import matrix
from warpack import CGInverse, CuDSSInverse, GeneralizedKrylovSchur, SparseOperator
from warpack.generalized import generalized_outputs, generalized_residual


@wp.kernel
def lift_normalize(
    source: wp.array2d[wp.float64],
    mu: wp.array[wp.float64],
    free: wp.array[wp.int32],
    out: wp.array2d[wp.float64],
):
    j, i = wp.tid()
    scale = wp.float64(wp.nan)
    if mu[j] > wp.float64(0.0) and wp.isfinite(mu[j]):
        scale = wp.float64(1.0) / wp.sqrt(mu[j])
    out[j, free[i]] = source[j, i] * scale


@wp.kernel
def boundary_means(
    v: wp.array2d[wp.float64], boundary: wp.array[wp.int32], means: wp.array2d[wp.float64], n: int
):
    j, i, c = wp.tid()
    wp.atomic_add(means, j, c, v[j, boundary[i] + c * n] / wp.float64(boundary.shape[0]))


@wp.kernel
def center(v: wp.array2d[wp.float64], means: wp.array2d[wp.float64], n: int):
    j, i = wp.tid()
    v[j, i] -= means[j, i // n]


@wp.kernel
def reciprocal(mu: wp.array[wp.float64], shift: wp.float64, values: wp.array[wp.float64]):
    j = wp.tid()
    values[j] = shift + wp.float64(1.0) / mu[j]


@wp.kernel
def scale_reciprocals(source: wp.array[wp.float64], scale: wp.float64, out: wp.array[wp.float64]):
    j = wp.tid()
    out[j] = scale * source[j]


class ConformalEigensolver:
    """Kx=lambda Mx -> Mx=mu (K-shift M)x; mu=1/(lambda-shift).

    The gauge and lift belong to this benchmark, not the eigensolver core.
    All replay arithmetic, including lifting and boundary centering, is Warp.
    """

    def __init__(self, case, ncv=32, backend="cudss", tol=1e-10, cg_tol=1e-11, implicit=False):
        self.case, self.k, self.tol = case, case.k, tol
        stiffness, mass = case.reduced
        shifted = matrix(stiffness - case.shift * mass)
        implicit_mass = implicit_shifted = None
        if implicit:
            from benchmarks.experimental.boundary_projector import (
                CenteredBoundary,
                ShiftedConformal,
            )

            implicit_mass = CenteredBoundary(case, ncv)
            implicit_shifted = ShiftedConformal(case, ncv)
        if backend == "cudss":
            inverse = CuDSSInverse(shifted, 1, refinement_steps=1)
        elif backend in ("cg", "cg-fsai"):
            preconditioner = linear.preconditioner(shifted, ptype="diag") if implicit else "diag"
            if backend == "cg-fsai":
                preconditioner = linear.FSAI(
                    shifted,
                    max_row_size=32,
                    max_step_size=3,
                    apply_lanes=8,
                    factor_dtype=wp.float32,
                )
            inverse = CGInverse(
                implicit_shifted.linear if implicit else shifted,
                tol=cg_tol,
                atol=0.0,
                maxiter=10000,
                preconditioner=preconditioner,
                verify_residuals=True,
                refinement_steps=1,
            )
        else:
            raise ValueError("unknown inverse backend")
        self.inverse = inverse
        self.device = shifted.device
        start = wp.array(
            np.random.default_rng(2026).normal(size=case.n), dtype=wp.float64, device=self.device
        )
        self.core = GeneralizedKrylovSchur(
            implicit_mass if implicit else SparseOperator(matrix(mass)),
            implicit_shifted if implicit else SparseOperator(shifted),
            inverse,
            case.k,
            ncv=ncv,
            tol=tol,
            mode="regular",
            which="LA",
            initial_vector=start,
        )
        self._allocate_outputs(case, self.core, inverse, tol)

    def _allocate_outputs(self, case, core, inverse, tol, mu_scale=1.0):
        """Shared original-pencil recovery; core vectors are A-normalized."""
        self.case, self.k, self.tol = case, case.k, tol
        self.core, self.inverse, self.device = core, inverse, core.device
        self.mu_scale = mu_scale
        self.mu = core.eigenvalues if mu_scale == 1.0 else wp.empty_like(core.eigenvalues)
        self.original, self.mass = (SparseOperator(matrix(m)) for m in (case.stiffness, case.mass))
        self.free = wp.array(case.free, dtype=wp.int32, device=self.device)
        self.boundary = wp.array(case.boundary, dtype=wp.int32, device=self.device)
        self.eigenvectors = wp.zeros(
            (case.k, case.stiffness.shape[0]), dtype=wp.float64, device=self.device
        )
        self.eigenvalues = wp.zeros(case.k, dtype=wp.float64, device=self.device)
        self.residuals = wp.empty_like(self.eigenvalues)
        self.converged = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.means = wp.zeros((case.k, 2), dtype=wp.float64, device=self.device)
        self.ax, self.bx = wp.empty_like(self.eigenvectors), wp.empty_like(self.eigenvectors)
        self.norms = wp.zeros((case.k, 3), dtype=wp.float64, device=self.device)
        self.order = wp.array(np.arange(case.k), dtype=wp.int32, device=self.device)

    def finalize(self):
        if self.mu_scale != 1.0:
            wp.launch(
                scale_reciprocals,
                self.k,
                [self.core.eigenvalues, self.mu_scale, self.mu],
                device=self.device,
            )
        self.eigenvectors.zero_()
        wp.launch(
            lift_normalize,
            self.core.eigenvectors.shape,
            [self.core.eigenvectors, self.mu, self.free, self.eigenvectors],
            device=self.device,
        )
        self.means.zero_()
        wp.launch(
            boundary_means,
            (self.k, self.boundary.size, 2),
            [self.eigenvectors, self.boundary, self.means, len(self.case.xyz)],
            device=self.device,
        )
        wp.launch(
            center,
            self.eigenvectors.shape,
            [self.eigenvectors, self.means, len(self.case.xyz)],
            device=self.device,
        )
        wp.launch(
            reciprocal,
            self.k,
            [self.mu, self.case.shift, self.eigenvalues],
            device=self.device,
        )
        self.original.apply(self.eigenvectors, self.ax)
        self.mass.apply(self.eigenvectors, self.bx)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (self.eigenvectors.shape[1] + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.bx, self.eigenvalues, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            generalized_outputs,
            self.k,
            [
                self.eigenvalues,
                self.order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

    def capture(self, steps):
        return self.core.capture(steps, finalizer=self.finalize)

    def close(self):
        self.core.graph = None
        if hasattr(self.inverse, "close"):
            self.inverse.close()
