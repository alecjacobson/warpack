"""Launch-size-only experiment for the existing sequential triangular kernel."""

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K


class TriangularBlockSize(BlockLOBPCG):
    _triangular_minimum = 96

    def __init__(self, *args, triangular_block_dim=64, **kwargs):
        if triangular_block_dim not in (32, 64, 128, 256, 512):
            raise ValueError("triangular block size must be32/64/128/256/512")
        self._triangular_block_dim = triangular_block_dim
        super().__init__(*args, **kwargs)

    def launch(self, kernel, shape, args):
        if kernel is K.triangular and self.b >= self._triangular_minimum:
            wp.launch(kernel, shape, args, device=self.device, block_dim=self._triangular_block_dim)
        else:
            super().launch(kernel, shape, args)
