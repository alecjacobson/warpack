"""Explicit benchmark controls for projected Cholesky dispatch."""

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K
from warpack._projected_cholesky import packed_cholesky_kernel


class SerialCholeskyBlock(BlockLOBPCG):
    """Force the historical factor even after the public default is optimized."""

    def launch(self, kernel, shape, args):
        if kernel is K.cholesky:
            wp.launch(kernel, shape, args, device=self.device)
            return
        super().launch(kernel, shape, args)


class CooperativeCholeskyBlock(BlockLOBPCG):
    """Benchmark-only substitution; every other block operation is unchanged."""

    def launch(self, kernel, shape, args):
        if kernel is K.cholesky:
            size = args[0].shape[0]
            required_shared = 8 * (size * (size + 1) // 2 + 1)
            if (
                self.device.is_cuda
                and 1 <= size <= 128
                and required_shared + 1024 <= self.device.max_shared_memory_per_block
            ):
                wp.launch_tiled(
                    packed_cholesky_kernel(size),
                    dim=1,
                    inputs=args,
                    block_dim=128,
                    device=self.device,
                )
                return
        super().launch(kernel, shape, args)
