"""Direct blocked triangular solve; no inverse and both QR passes retained.

Each block owns disjoint physical columns. Completed row panels are reused by
small tiled matrix products; the diagonal panel uses ordered forward substitution.
This changes summation order and requires original-equation validation.
"""

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K


@wp.kernel(enable_backward=False)
def blocked_triangular(q: wp.array2d[wp.float64], factor: wp.array2d[wp.float64]):
    block, lane = wp.tid()
    for panel in range((q.shape[0] + 7) // 8):
        product = wp.tile_zeros(shape=(8, 32), dtype=wp.float64)
        for previous in range(panel):
            a = wp.tile_load(factor, shape=(8, 8), offset=(panel * 8, previous * 8))
            b = wp.tile_load(q, shape=(8, 32), offset=(previous * 8, block * 32))
            wp.tile_matmul(a, b, product)
        rhs = wp.tile_load(q, shape=(8, 32), offset=(panel * 8, block * 32))
        residual = rhs - product
        values = wp.tile_zeros(shape=256, dtype=wp.float64, storage="shared")
        wp.tile_assign(values, wp.tile_reshape(residual, shape=(256,)))
        for row in range(8):
            absolute = panel * 8 + row
            value = wp.tile_extract(values, row * 32 + lane)
            if absolute < q.shape[0]:
                for previous in range(row):
                    value -= factor[absolute, panel * 8 + previous] * wp.tile_extract(
                        values, previous * 32 + lane
                    )
                value /= factor[absolute, absolute]
            wp.tile_scatter_masked(values, row * 32 + lane, value, True)
        wp.tile_store(q, wp.tile_reshape(values, shape=(8, 32)), offset=(panel * 8, block * 32))
        # Complete all global stores before this block loads the solved panel in
        # the next iteration. Blocks never share physical columns.
        wp.tile_scatter_masked(values, lane, wp.float64(0.0), True)


class BlockedTriangularBlock(BlockLOBPCG):
    def launch(self, kernel, shape, args):
        if kernel is K.triangular:
            wp.launch_tiled(
                blocked_triangular,
                dim=(args[0].shape[1] + 31) // 32,
                inputs=args,
                block_dim=32,
                device=self.device,
            )
        else:
            super().launch(kernel, shape, args)
