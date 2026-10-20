"""Compatibility settings adapter for the recorded CG correction experiment.

The original algorithm prototype is preserved in commit aac7b5a. Numerical
correction now belongs to the public CGInverse implementation.
"""

from warpack import CGInverse


class CorrectedCGInverse(CGInverse):
    def __init__(
        self,
        matrix,
        tol=1e-11,
        correction_tol=0.01,
        correction_steps=1,
        maxiter=10000,
        preconditioner=None,
    ):
        super().__init__(
            matrix,
            tol=tol,
            atol=0.0,
            maxiter=maxiter,
            preconditioner=preconditioner,
            verify_residuals=True,
            refinement_steps=correction_steps,
            refinement_tol=correction_tol,
        )
