"""Explicit full normal-space projection, avoiding serial Krylov expansion.

An experimental control restricted to ncv == min(A.shape). It applies the
normal operator to the entire coordinate basis in one batch, then uses the
existing Warp projected eigensolve and guarded SVD finalization. It forms the
small projected normal matrix explicitly; no partial-space shortcut is made.
"""

import warp as wp

from warpack import NormalSVD, SymmetricEigensolver


@wp.kernel
def coordinate_basis(q: wp.array2d[wp.float64]):
    i, j = wp.tid()
    q[i, j] = wp.float64(0.0)
    if i == j:
        q[i, j] = wp.float64(1.0)


class _FullNormalCore(SymmetricEigensolver):
    def initialize(self, seed=42):
        self.op.refresh()
        self.status.zero_()
        self.recoveries.zero_()
        wp.launch(coordinate_basis, self.q.shape, [self.q], device=self.device)

    def step(self):
        # The full coordinate basis already spans the whole normal space.
        # The base finalizer computes its products, projection and Ritz pairs.
        pass


class FullBasisNormalSVD(NormalSVD):
    def __init__(self, matrix, k, ncv=None, **kwargs):
        dimension = min(matrix.shape)
        if ncv is not None and ncv != dimension:
            raise ValueError("full normal projection requires ncv == min(A.shape)")
        super().__init__(matrix, k, ncv=dimension, **kwargs)
        self.core = _FullNormalCore(self.op, k, ncv=dimension, which="LA", tol=self.tol)
        self.core.recoveries = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.status = self.core.status
