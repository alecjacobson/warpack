"""Experimental final column normalization for accumulated Jacobi norm drift."""

import warp as wp

from warpack.projected import GlobalJacobi
from warpack.projected import normalize_eigenvectors as normalize_columns


class LegacyJacobi(GlobalJacobi):
    _normalize_vectors = False


class NormalizedJacobi(LegacyJacobi):
    # Explicit all-size prototype; suppress the installed wide-only addition
    # so this control applies normalization exactly once.
    def _capture(self):
        super()._capture()
        wp.launch_tiled(
            normalize_columns, dim=self.n, inputs=[self.v], block_dim=128, device=self.device
        )
