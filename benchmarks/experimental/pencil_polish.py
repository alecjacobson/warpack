"""Explicit postprocessing control for a generalized shift-invert pencil.

One application of (A-sigma B)^-1 B preserves exact right modes and filters
off-target vector error. All backends must receive the same declared polish
count; original pencil/quadratic/count/subspace checks still decide acceptance.
This benchmark adapter does not change the public algorithm or defaults.
"""

import warp as wp

from warpack.pencil import GeneralizedShiftInvertEigensolver

from .damped_coordinates import normalize, row_norms


class PolishedPencil(GeneralizedShiftInvertEigensolver):
    def __init__(self, *args, polish_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.polish_steps = polish_steps
        self.polished = [wp.empty_like(self.core.eigenvectors) for _ in range(min(2, polish_steps))]
        self.polish_norms = None
        if polish_steps:
            self.polish_norms = wp.zeros((self.k, 1), dtype=wp.float64, device=self.device)
            self.eigenvectors = wp.empty_like(self.core.eigenvectors)

    def finalize(self):
        source = self.core.eigenvectors
        for step in range(self.polish_steps):
            result = self.polished[step % len(self.polished)]
            self.transform.apply_complex(source, result)
            self.polish_norms.zero_()
            wp.launch_tiled(
                row_norms,
                dim=(self.k, (self.op.complex_n + 255) // 256),
                inputs=[result, self.polish_norms],
                block_dim=128,
                device=self.device,
            )
            wp.launch(normalize, result.shape, [result, self.polish_norms], device=self.device)
            source = result
        if self.polish_steps:
            wp.copy(self.eigenvectors, source)
        super().finalize()
