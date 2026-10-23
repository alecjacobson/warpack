"""Cache solved columns in shared memory while retaining serial subtraction order."""

import functools

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K


@functools.lru_cache(None)
def shared_kernel(size):
    if not 1 <= size <= 256:
        raise ValueError("shared triangular solve supports widths 1 through 256")
    count = wp.constant(size * 32)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(q: wp.array2d[wp.float64], factor: wp.array2d[wp.float64]):
        block, lane = wp.tid()
        column = block * 32 + lane
        solved = wp.tile_zeros(shape=count, dtype=wp.float64, storage="shared")
        for row in range(q.shape[0]):
            value = wp.float64(0.0)
            if column < q.shape[1]:
                value = q[row, column]
                for previous in range(row):
                    value -= factor[row, previous] * wp.tile_extract(solved, previous * 32 + lane)
                value /= factor[row, row]
            # Each lane owns its column; the collective also keeps all lanes
            # aligned before the next row. Inactive tail lanes remain masked.
            wp.tile_scatter_masked(solved, row * 32 + lane, value, column < q.shape[1])
            if column < q.shape[1]:
                q[row, column] = value

    return compute


class SharedTriangularBlock(BlockLOBPCG):
    def launch(self, kernel, shape, args):
        if kernel is K.triangular and args[0].shape[0] <= 256:
            wp.launch_tiled(
                shared_kernel(args[0].shape[0]),
                dim=(args[0].shape[1] + 31) // 32,
                inputs=args,
                block_dim=32,
                device=self.device,
            )
        else:
            super().launch(kernel, shape, args)
