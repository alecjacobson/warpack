"""Structure-aware inverse application, retaining both CholeskyQR passes."""

import warp as wp

from benchmarks.experimental.triangular_rotate import TriangularRotateBlock


@wp.kernel(enable_backward=False)
def lower_rotate(
    q: wp.array2d[wp.float64], coeff: wp.array2d[wp.float64], out: wp.array2d[wp.float64]
):
    i, j = wp.tid()
    acc = wp.tile_zeros(shape=(4, 32), dtype=wp.float64)
    # Every coefficient beyond the final row of this tile is exactly zero:
    # inverse_lower explicitly writes the full upper triangle on every replay.
    stop = wp.min(q.shape[0], (i + 1) * 4)
    for block in range((stop + 7) // 8):
        a = wp.tile_load(coeff, shape=(4, 8), offset=(i * 4, block * 8))
        b = wp.tile_load(q, shape=(8, 32), offset=(block * 8, j * 32))
        wp.tile_matmul(a, b, acc)
    wp.tile_store(out, acc, offset=(i * 4, j * 32))


class LowerTriangularRotateBlock(TriangularRotateBlock):
    def _apply_inverse(self, source, out):
        wp.launch_tiled(
            lower_rotate,
            dim=((out.shape[0] + 3) // 4, (out.shape[1] + 31) // 32),
            inputs=[source, self.gx, out],
            block_dim=128,
            device=self.device,
        )
