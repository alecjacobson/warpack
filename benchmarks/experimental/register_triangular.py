"""Eight serial RHS rows per thread, reusing earlier values in registers.

Each row retains the original subtraction order. Interleaving eight independent
recurrences reuses each earlier solved entry without matrix-product regrouping,
a triangular inverse, shared-memory barriers or additional global workspace.
"""

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K

vec8d = wp.types.vector(length=8, dtype=wp.float64)


@wp.kernel(enable_backward=False)
def register_triangular(q: wp.array2d[wp.float64], factor: wp.array2d[wp.float64]):
    column = wp.tid()
    for panel in range((q.shape[0] + 7) // 8):
        rhs = vec8d()
        # Constant loops of length8 (and their constant diagonal prefixes)
        # unroll, keeping component indices static instead of spilling a
        # dynamically indexed full-width local array.
        for row in range(8):
            absolute = panel * 8 + row
            if absolute < q.shape[0]:
                rhs[row] = q[absolute, column]
        for previous in range(panel * 8):
            solved = q[previous, column]
            for row in range(8):
                absolute = panel * 8 + row
                if absolute < q.shape[0]:
                    rhs[row] -= factor[absolute, previous] * solved
        for row in range(8):
            absolute = panel * 8 + row
            if absolute < q.shape[0]:
                value = rhs[row]
                for previous in range(row):
                    value -= factor[absolute, panel * 8 + previous] * rhs[previous]
                rhs[row] = value / factor[absolute, absolute]
        for row in range(8):
            absolute = panel * 8 + row
            if absolute < q.shape[0]:
                q[absolute, column] = rhs[row]


class RegisterTriangularBlock(BlockLOBPCG):
    def launch(self, kernel, shape, args):
        if kernel is K.triangular:
            wp.launch(
                register_triangular,
                args[0].shape[1],
                inputs=args,
                block_dim=256,
                device=self.device,
            )
        else:
            super().launch(kernel, shape, args)
