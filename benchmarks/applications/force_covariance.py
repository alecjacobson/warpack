"""Implicit force-covariance composition; application code outside the solver core.

For SPD H/M and symmetric PSD force prior S, solve the dominant generalized
pencil (M H^-1 S H^-1 M, M). Neither an n-by-n displacement covariance nor a
full set of response snapshots is constructed on device.
"""

import numpy as np
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from warpack import CGInverse, CuDSSInverse, GeneralizedKrylovSchur, SparseOperator


class ForceCovarianceOperator:
    def __init__(self, mass, prior, inverse):
        if any(op.n != mass.n or op.device != mass.device for op in (prior, inverse)):
            raise ValueError("mass, force prior and inverse must share dimensions/device")
        self.mass, self.prior, self.inverse = mass, prior, inverse
        self.n, self.device = mass.n, mass.device
        self.requires_static_outer_capture = any(
            getattr(op, "requires_static_outer_capture", False) for op in (mass, prior, inverse)
        )
        self.left = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        self.right = wp.empty_like(self.left)

    def apply(self, x, y):
        # Shapes select a fixed capture-time sequence, never numerical decisions.
        # The solver supplies disjoint arrays; exact in-place rows also work.
        if x.ndim != 2 or x.shape != y.shape or x.shape[1] != self.n:
            raise ValueError("expected matching (width,n) rows")
        for j in range(x.shape[0]):
            self.mass.apply(x[j : j + 1], self.left)
            self.inverse.apply(self.left, self.right)
            self.prior.apply(self.right, self.left)
            self.inverse.apply(self.left, self.right)
            self.mass.apply(self.right, y[j : j + 1])


class ForceCovarianceSolver:
    """Fixed H/M, mutable prior values/topology fixed; positive covariance modes.

    Sparse inputs are device matrices. SPD H/M and PSD prior are caller
    preconditions; original equations and requested subspaces need validation.
    The force prior is never inverted, including for singular PSD priors.
    """

    def __init__(
        self, h, mass, prior, k, backend="cudss", *, metric_backend=None, ncv=32, tol=1e-11
    ):
        if h.shape != mass.shape or prior.shape != h.shape:
            raise ValueError("matching matrix dimensions required")
        self.backends = (backend, backend if metric_backend is None else metric_backend)
        if any(x not in ("cudss", "cg") for x in self.backends):
            raise ValueError("backend must be cudss or cg")
        self.device = h.device
        self.factors = []
        self.inverses = []
        try:
            for source, kind in zip((h, mass), self.backends):
                if kind == "cudss":
                    inverse = CuDSSInverse(source, 1, mtype="spd")
                else:
                    factor = linear.FSAI(
                        source, max_row_size=32, max_step_size=3, factor_dtype=wp.float32
                    )
                    self.factors.append(factor)
                    inverse = CGInverse(
                        source,
                        preconditioner=factor,
                        tol=1e-13,
                        atol=0,
                        maxiter=2000,
                        adaptive_capture=True,
                        verify_residuals=True,
                    )
                self.inverses.append(inverse)
            self.operator = ForceCovarianceOperator(
                SparseOperator(mass), SparseOperator(prior), self.inverses[0]
            )
            self.solver = GeneralizedKrylovSchur(
                self.operator,
                SparseOperator(mass),
                self.inverses[1],
                k,
                ncv=min(ncv, h.shape[0]),
                mode="regular",
                which="LA",
                tol=tol,
            )
        except Exception:
            for inverse in self.inverses:
                if hasattr(inverse, "close"):
                    inverse.close()
            raise
        self.graph = None

    def capture(self, cycles=4, *, adaptive=None):
        self.adaptive = (
            not self.solver.iter_op.requires_static_outer_capture if adaptive is None else adaptive
        )
        self.graph = self.solver.capture(cycles, adaptive=adaptive)
        # Complete first graph upload during setup; changed-prior replay tests
        # must validate the first caller solve, not just these initialized values.
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        return self.graph

    def solve(self):
        if self.graph is None:
            raise ValueError("capture before replay")
        for inverse in self.inverses:
            if hasattr(inverse, "reset_stats"):
                inverse.reset_stats()
        wp.capture_launch(self.graph)

    def close(self):
        self.graph = self.solver.graph = None
        for inverse in self.inverses:
            if hasattr(inverse, "close"):
                inverse.close()


def dense_reference(h, mass, prior, k):
    """Independent small CPU oracle only; never used in GPU iteration."""
    if h.shape[0] > 2048:
        raise ValueError("dense reference is limited to small validation cases")
    hd, md, sd = (x.toarray() for x in (h, mass, prior))
    inv = la.cho_solve(la.cho_factor(hd), np.eye(len(hd)))
    a = md @ inv @ sd @ inv @ md
    a = (a + a.T) * 0.5
    values, vectors = la.eigh(a, md)
    return dict(a=a, values=values[-k:][::-1].copy(), basis=vectors[:, -k:][:, ::-1].copy())
