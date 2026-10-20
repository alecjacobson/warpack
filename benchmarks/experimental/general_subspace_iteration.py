"""Fixed-width orthogonal iteration diagnostic for direct largest-magnitude modes.

Preserve the entire m-dimensional search space instead of Ritz truncation. Each
cycle applies A to every basis vector. Only LM is supported by this experiment.
"""

import warp as wp

from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.arnoldi import complex_gram, rotate_complex, scale_vector


class SubspaceIterationGeneral(StartedGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.which != 0:
            raise ValueError("Subspace iteration experiment requires LM")

    def initialize(self, seed=42):
        super().initialize(seed)
        for j in range(self.keep, self.m):
            self.expand(j)

    def step(self):
        self.op.apply_complex(self.q, self.z)
        wp.copy(self.q, self.z)
        for j in range(self.m):
            self.orthogonalize(j)
            wp.launch(scale_vector, self.n, [self.qrows[j], self.norm], device=self.device)
        self.op.apply_complex(self.q, self.z)
        self.h.zero_()
        wp.launch_tiled(
            complex_gram,
            dim=(self.m, self.m, (self.n + 255) // 256),
            inputs=[self.q, self.z, self.h],
            block_dim=128,
            device=self.device,
        )
        self.diagonalize()
        self.select()
        wp.launch(
            rotate_complex,
            self.eigenvectors.shape,
            [self.q, self.projected, self.order, self.eigenvectors],
            device=self.device,
        )
        self.prepare_restart()

    def capture(self, max_iterations=100, adaptive=False, finalizer=None):
        if adaptive is not False:
            raise ValueError("Subspace iteration experiment requires a fixed budget")
        return super().capture(max_iterations, adaptive=False, finalizer=finalizer)
