"""Invertible force-prior recipe: implicit reciprocal covariance pencil.

Q=H S^-1 H, Q^-1=H^-1 S H^-1. Both compositions remain implicit, and only H
needs factorization/preconditioning. H, M, S and the supplied S^-1 must be SPD.
"""

import warp as wp
from warp.optim import linear

from warpack import (
    CGInverse,
    CuDSSInverse,
    GeneralizedEigensolver,
    GeneralizedKrylovSchur,
    SparseOperator,
)


@wp.kernel
def reciprocal_values(mu: wp.array[wp.float64], values: wp.array[wp.float64]):
    i = wp.tid()
    values[i] = wp.float64(1.0) / mu[i]


class SandwichOperator:
    """Symmetric outer @ middle @ outer, using fixed scratch and rowwise calls."""

    def __init__(self, outer, middle):
        if outer.n != middle.n or outer.device != middle.device:
            raise ValueError("matching dimensions/device required")
        self.outer, self.middle = outer, middle
        self.n, self.device = outer.n, outer.device
        self.requires_static_outer_capture = any(
            getattr(op, "requires_static_outer_capture", False) for op in (outer, middle)
        )
        self.left = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        self.right = wp.empty_like(self.left)

    def apply(self, x, y):
        if x.ndim != 2 or x.shape != y.shape or x.shape[1] != self.n:
            raise ValueError("expected matching (width,n) rows")
        for j in range(x.shape[0]):
            self.outer.apply(x[j : j + 1], self.left)
            self.middle.apply(self.left, self.right)
            self.outer.apply(self.right, y[j : j + 1])


class ForceReciprocalSolver:
    """Fixed H/M, matched mutable S/S^-1 values at fixed storage/topology.

    Inputs are device sparse matrices. Positive definiteness and consistency
    of S/S^-1 are caller preconditions, verified independently in the tests.
    Singular priors must use the covariance/snapshot recipes instead.
    """

    def __init__(
        self,
        h,
        mass,
        prior,
        prior_inverse,
        k,
        backend="cudss",
        *,
        ncv=64,
        tol=1e-9,
        refinement_steps=0,
        polish_steps=0,
    ):
        if any(a.shape != h.shape or a.device != h.device for a in (mass, prior, prior_inverse)):
            raise ValueError("matching matrix dimensions/device required")
        if backend not in ("cudss", "cg"):
            raise ValueError("backend must be cudss or cg")
        if backend == "cudss" and refinement_steps != 0:
            raise ValueError("refinement_steps configures CG only")
        if isinstance(polish_steps, bool) or not isinstance(polish_steps, int) or polish_steps < 0:
            raise ValueError("polish_steps must be a nonnegative integer")
        self.polish_steps = polish_steps
        self.device = h.device
        self.factor = None
        if backend == "cudss":
            self.inverse = CuDSSInverse(h, 1, mtype="spd")
        else:
            self.factor = linear.FSAI(h, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
            self.inverse = CGInverse(
                h,
                preconditioner=self.factor,
                tol=1e-13,
                atol=0,
                maxiter=2000,
                adaptive_capture=True,
                verify_residuals=True,
                refinement_steps=refinement_steps,
            )
        try:
            self.operator = SandwichOperator(SparseOperator(h), SparseOperator(prior_inverse))
            self.covariance = SandwichOperator(self.inverse, SparseOperator(prior))
            self.krylov = GeneralizedKrylovSchur(
                self.operator,
                SparseOperator(mass),
                self.covariance,
                k,
                ncv=min(ncv, h.shape[0]),
                mode="shift-invert",
                target=0,
                tol=tol,
            )
            self.polisher = (
                GeneralizedEigensolver(
                    self.operator,
                    SparseOperator(mass),
                    self.covariance,
                    k,
                    ncv=k,
                    tol=tol,
                    initial_vectors=self.krylov.eigenvectors,
                )
                if polish_steps
                else None
            )
            self.solver = self.polisher or self.krylov
            self.eigenvalues = wp.empty(k, dtype=wp.float64, device=self.device)
            self.eigenvectors = self.solver.eigenvectors
        except Exception:
            if hasattr(self.inverse, "close"):
                self.inverse.close()
            raise
        self.graph = None
        self.polish_graph = None
        self.recovery_graph = None

    def capture(self, cycles=4, *, adaptive=None):
        """Initialize and return the ordered graphs; ``solve`` replays all stages."""
        self.adaptive = (
            not self.krylov.iter_op.requires_static_outer_capture if adaptive is None else adaptive
        )
        self.graph = self.krylov.capture(cycles, adaptive=adaptive)
        # Finish first inner-graph upload before composing dependent recovery.
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        if self.polisher:
            self.polish_graph = self.polisher.capture(self.polish_steps, adaptive=False)
            wp.capture_launch(self.polish_graph)
            wp.synchronize_device(self.device)
        wp.launch(
            reciprocal_values,
            self.eigenvalues.size,
            [self.solver.eigenvalues, self.eigenvalues],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(
                reciprocal_values,
                self.eigenvalues.size,
                [self.solver.eigenvalues, self.eigenvalues],
                device=self.device,
            )
        self.recovery_graph = capture.graph
        wp.capture_launch(self.recovery_graph)
        wp.synchronize_device(self.device)
        return tuple(
            g for g in (self.graph, self.polish_graph, self.recovery_graph) if g is not None
        )

    def solve(self):
        if self.graph is None:
            raise ValueError("capture before replay")
        if hasattr(self.inverse, "reset_stats"):
            self.inverse.reset_stats()
        wp.capture_launch(self.graph)
        if self.polish_graph:
            wp.capture_launch(self.polish_graph)
        wp.capture_launch(self.recovery_graph)

    def close(self):
        self.graph = self.solver.graph = None
        self.krylov.graph = self.polish_graph = None
        self.recovery_graph = None
        if hasattr(self.inverse, "close"):
            self.inverse.close()
