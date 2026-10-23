"""Experiment: tiled applications of a small triangular inverse in both X passes.

No production dispatch changes. Both complete CholeskyQR passes, scaling,
cached operator-image transformations and final mass application are retained.
The changed floating arithmetic needs independent physical validation.
"""

import warp as wp

from benchmarks.experimental.first_metric_certificate import inverse_lower
from warpack import BlockLOBPCG
from warpack import kernels as K
from warpack.block import normalize_gram, rotate, scale_rows, scales


class TriangularRotateBlock(BlockLOBPCG):
    _inverse_minimum = 96

    def _apply_inverse(self, source, out):
        rotate(source, self.gx, out)

    def orthogonalize_x(self):
        if self.b < self._inverse_minimum:
            return super().orthogonalize_x()
        for i in range(2):
            self.mass.apply(self.sx, self.bx)
            self.gram(self.sx, self.bx, self.gx, symmetric=True)
            if i == 0:
                self.launch(scales, self.b, [self.gx, self.xscale, self.status])
                self.launch(scale_rows, self.sx.shape, [self.sx, self.xscale])
                if self._image_reuse_active:
                    self.launch(scale_rows, self.sx.shape, [self._image_x, self.xscale])
                self.launch(normalize_gram, self.gx.shape, [self.gx, self.xscale])
            self.launch(K.cholesky, 1, [self.gx, self.lx, self.status])
            # gx is dead after factorization. bx is dead after forming gx;
            # reuse it sequentially for X and cached AX before refreshing M X.
            wp.launch(
                inverse_lower,
                self.b,
                [self.lx, self.gx, self.status],
                block_dim=32,
                device=self.device,
            )
            self._apply_inverse(self.sx, self.bx)
            wp.copy(self.sx, self.bx)
            if self._image_reuse_active:
                self._apply_inverse(self._image_x, self.bx)
                wp.copy(self._image_x, self.bx)
        self.mass.apply(self.sx, self.bx)
