"""Independent largest-magnitude reference, including conjugate boundary ties."""

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment


class GeneralReference:
    def __init__(self, a, k, margin=1e-8):
        values, basis = la.eig(a)
        cutoff = np.sort(abs(values))[-k]
        self.a, self.k, self.margin = a, k, margin
        self.allowed = np.flatnonzero(abs(values) >= cutoff - margin)
        self.required = set(np.flatnonzero(abs(values) > cutoff + margin))
        self.values = values[self.allowed]
        self.basis = la.orth(basis[:, self.allowed])

    def check(self, values, vectors):
        if (
            values.shape != (self.k,)
            or vectors.shape != (len(self.a), self.k)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(vectors))
        ):
            return dict(accepted=False, reason="Invalid output shape or nonfinite eigenpairs")
        cost = abs(values[:, None] - self.values)
        rows, cols = linear_sum_assignment(cost)
        error = self.a @ vectors - vectors * values
        result = dict(
            max_absolute_residual=float(abs(error).max()),
            max_normalized_residual=float(
                np.max(np.linalg.norm(error, axis=0) / np.maximum(1, abs(values)))
            ),
            spectrum_error=float(cost[rows, cols].max()),
            required_spectrum_covered=self.required.issubset(self.allowed[cols]),
            vector_norm_error=float(abs(np.linalg.norm(vectors, axis=0) - 1).max()),
            subspace_error=float(
                np.linalg.norm(vectors - self.basis @ (self.basis.conj().T @ vectors), 2)
            ),
        )
        result["accepted"] = bool(
            result["required_spectrum_covered"]
            and result["max_absolute_residual"] < self.margin
            and result["spectrum_error"] < self.margin
            and result["vector_norm_error"] < self.margin
            and result["subspace_error"] < self.margin
        )
        return result
