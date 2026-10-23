"""Benchmark override for column-cooperative Cholesky with global storage.

All arithmetic remains Warp. A transposed device workspace coalesces the
per-column row accesses; shared tile collectives synchronize each completed
column before dependent global reads. Production guards this kernel to wider
blocks; the benchmark override can force it at any supported dimension.
"""

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K
from warpack._projected_cholesky import global_cholesky


class GlobalCholeskyBlock(BlockLOBPCG):
    """Benchmark-only substitution, including reusable workspace in setup."""

    def __init__(self, *args, **kwargs):
        # Active-search branches shallow-copy the owner during super().__init__.
        # A shared holder gives them the workspace without allocating in replay.
        self._factor_workspace = [None]
        super().__init__(*args, **kwargs)
        if self.device.is_cuda and 1 <= self.b <= 512:
            self._factor_workspace[0] = (
                self._global_factor_workspace
                if self._global_factor_workspace is not None
                else wp.empty((self.b, self.b), dtype=wp.float64, device=self.device)
            )

    def launch(self, kernel, shape, args):
        if kernel is K.cholesky and self.device.is_cuda:
            size = args[0].shape[0]
            if 1 <= size <= 512:
                wp.launch_tiled(
                    global_cholesky,
                    dim=1,
                    inputs=[*args, self._factor_workspace[0]],
                    block_dim=256 if size <= 256 else 512,
                    device=self.device,
                )
                return
        super().launch(kernel, shape, args)
