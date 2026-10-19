"""Explicit complete coordinate projection beyond the public small-space policy.

The installed API uses this numerical core automatically for complete spaces
of at most 64 dimensions. This benchmark adapter additionally allows larger
complete spaces; it is tied to private core interfaces in this checkout.
"""

from warpack import NormalSVD
from warpack.normal_svd import _FullNormalCore


class FullBasisNormalSVD(NormalSVD):
    def __init__(self, matrix, k, ncv=None, **kwargs):
        dimension = min(matrix.shape)
        if ncv is not None and ncv != dimension:
            raise ValueError("full normal projection requires ncv == min(A.shape)")
        super().__init__(matrix, k, ncv=dimension, **kwargs)
        if not isinstance(self.core, _FullNormalCore):
            self.core = _FullNormalCore(self.op, k, ncv=dimension, which="LA", tol=self.tol)
            self.status = self.core.status
