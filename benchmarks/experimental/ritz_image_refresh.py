"""Measure fresh A*X against two triangular transforms of the cached image.

Both complete mass QR passes remain. This benchmark-only choice can save work
for sparse operators at wide block sizes; an expensive implicit operator may
instead make it much slower. No installed dispatch or default changes here.
"""

from warpack import BlockLOBPCG


class FreshRitzImageBlock(BlockLOBPCG):
    def orthogonalize_x(self):
        cached = self._image_reuse_active
        self._image_reuse_active = False
        try:
            super().orthogonalize_x()
        finally:
            self._image_reuse_active = cached
        if cached:
            self.op.apply(self.sx, self._image_x)
