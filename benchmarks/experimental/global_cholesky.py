"""Experimental column-cooperative Cholesky without a shared-factor capacity limit.

All arithmetic remains Warp. A transposed device workspace coalesces the
per-column row accesses; shared tile collectives synchronize each completed
column before dependent global reads. No production dispatch is changed.
"""

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K


@wp.kernel(enable_backward=False)
def global_cholesky(
    g: wp.array2d[wp.float64],
    factor: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    transposed: wp.array2d[wp.float64],
):
    block, lane = wp.tid()
    n = g.shape[0]
    diagonal = wp.tile_zeros(shape=1, dtype=wp.float64, storage="shared")
    for j in range(n):
        i = j + lane
        value = wp.float64(0.0)
        if i < n:
            value = g[i, j]
            for k in range(j):
                value -= transposed[k, i] * transposed[k, j]
        root = wp.float64(0.0)
        if lane == 0:
            if not wp.isfinite(value) or value <= wp.float64(0.0):
                wp.atomic_max(status, 0, 1)
            root = wp.sqrt(wp.max(value, wp.float64(1.0e-30)))
        wp.tile_scatter_masked(diagonal, 0, root, lane == 0)
        root = wp.tile_extract(diagonal, 0)
        entry = value / root
        if lane == 0:
            entry = root
        if i < n:
            factor[i, j] = entry
            transposed[j, i] = entry
        # tile_scatter_masked is a block collective ending in WP_TILE_SYNC.
        # Its barrier also orders the global workspace writes above before the
        # next column reads them. Keep this second collective after the writes.
        wp.tile_scatter_masked(diagonal, 0, root, lane == 0)


class GlobalCholeskyBlock(BlockLOBPCG):
    """Benchmark-only substitution, including reusable workspace in setup."""

    def __init__(self, *args, **kwargs):
        # Active-search branches shallow-copy the owner during super().__init__.
        # A shared holder gives them the workspace without allocating in replay.
        self._factor_workspace = [None]
        super().__init__(*args, **kwargs)
        if self.device.is_cuda and 1 <= self.b <= 512:
            self._factor_workspace[0] = wp.empty(
                (self.b, self.b), dtype=wp.float64, device=self.device
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
