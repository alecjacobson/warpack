"""Benchmark-only panel-size controls for direct triangular substitution."""

import functools

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K


@functools.lru_cache(None)
def panel_kernel(panel_rows):
    if panel_rows not in (8, 16, 32):
        raise ValueError("panel_rows must be8,16or32")
    rows = wp.constant(panel_rows)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(q: wp.array2d[wp.float64], factor: wp.array2d[wp.float64]):
        block, lane = wp.tid()
        for panel in range((q.shape[0] + rows - 1) // rows):
            product = wp.tile_zeros(shape=(rows, 32), dtype=wp.float64)
            for previous in range(panel):
                a = wp.tile_load(factor, shape=(rows, rows), offset=(panel * rows, previous * rows))
                b = wp.tile_load(q, shape=(rows, 32), offset=(previous * rows, block * 32))
                wp.tile_matmul(a, b, product)
            rhs = wp.tile_load(q, shape=(rows, 32), offset=(panel * rows, block * 32))
            residual = rhs - product
            values = wp.tile_zeros(shape=wp.static(rows * 32), dtype=wp.float64, storage="shared")
            wp.tile_assign(values, wp.tile_reshape(residual, shape=(wp.static(rows * 32),)))
            for row in range(rows):
                absolute = panel * rows + row
                value = wp.tile_extract(values, row * 32 + lane)
                if absolute < q.shape[0]:
                    for previous in range(row):
                        value -= factor[absolute, panel * rows + previous] * wp.tile_extract(
                            values, previous * 32 + lane
                        )
                    value /= factor[absolute, absolute]
                wp.tile_scatter_masked(values, row * 32 + lane, value, True)
            wp.tile_store(
                q, wp.tile_reshape(values, shape=(rows, 32)), offset=(panel * rows, block * 32)
            )
            # Complete all global stores before this block loads the solved panel in
            # the next iteration. Blocks never share physical columns.
            wp.tile_scatter_masked(values, lane, wp.float64(0.0), True)

    return compute


class PanelTriangularBlock(BlockLOBPCG):
    def __init__(self, *args, panel_rows=16, **kwargs):
        self._panel_kernel = panel_kernel(panel_rows)
        super().__init__(*args, **kwargs)

    def launch(self, kernel, shape, args):
        if kernel is K.triangular:
            wp.launch_tiled(
                self._panel_kernel,
                dim=(args[0].shape[1] + 31) // 32,
                inputs=args,
                block_dim=32,
                device=self.device,
            )
        else:
            super().launch(kernel, shape, args)
