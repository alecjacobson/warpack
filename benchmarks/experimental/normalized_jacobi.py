"""Experimental final column normalization for accumulated Jacobi norm drift."""

import warp as wp

from warpack.projected import GlobalJacobi


@wp.kernel(enable_backward=False)
def normalize_columns(v: wp.array2d[wp.float64]):
    column, lane = wp.tid()
    square = wp.float64(0.0)
    for row in range(lane, v.shape[0], 128):
        value = v[row, column]
        square += value * value
    square = wp.tile_extract(wp.tile_sum(wp.tile(square)), 0)
    # Preserve invalid outputs for existing failure detection. Never turn a
    # nonfinite/zero vector into a plausible new eigenvector.
    if wp.isfinite(square) and square > wp.float64(0.0):
        inverse = wp.float64(1.0) / wp.sqrt(square)
        for row in range(lane, v.shape[0], 128):
            v[row, column] *= inverse


class NormalizedJacobi(GlobalJacobi):
    def _capture(self):
        super()._capture()
        wp.launch_tiled(
            normalize_columns, dim=self.n, inputs=[self.v], block_dim=128, device=self.device
        )
