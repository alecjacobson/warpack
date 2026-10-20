"""Explicit postprocessing control for a generalized shift-invert pencil.

One application of (A-sigma B)^-1 B preserves exact right modes and filters
off-target vector error. All backends must receive the same declared polish
count; original pencil/quadratic/count/subspace checks still decide acceptance.
This benchmark adapter does not change the public algorithm or defaults.
"""

import warp as wp

from warpack.hermitian import cdot, norm2
from warpack.pencil import (
    GeneralizedShiftInvertEigensolver,
    pencil_errors,
    pencil_norms,
    pencil_outputs,
)

from .damped_coordinates import normalize, row_norms


@wp.kernel
def quotient_parts(
    ax: wp.array2d[wp.vec2d],
    bx: wp.array2d[wp.vec2d],
    numerator: wp.array2d[wp.vec2d],
    denominator: wp.array2d[wp.float64],
):
    j, block = wp.tid()
    a = wp.tile_load(ax[j], shape=256, offset=block * 256)
    b = wp.tile_load(bx[j], shape=256, offset=block * 256)
    wp.tile_atomic_add(numerator[j], wp.tile_sum(wp.tile_map(cdot, b, a)))
    wp.tile_atomic_add(denominator[j], wp.tile_sum(wp.tile_map(norm2, b)))


@wp.kernel
def quotient_values(
    numerator: wp.array2d[wp.vec2d],
    denominator: wp.array2d[wp.float64],
    values: wp.array[wp.vec2d],
):
    j = wp.tid()
    values[j] = numerator[j, 0] / denominator[j, 0]


class PolishedPencil(GeneralizedShiftInvertEigensolver):
    def __init__(self, *args, polish_steps=0, rayleigh=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.polish_steps = polish_steps
        self.rayleigh = rayleigh
        self.polished = [wp.empty_like(self.core.eigenvectors) for _ in range(min(2, polish_steps))]
        self.polish_norms = None
        if polish_steps:
            self.polish_norms = wp.zeros((self.k, 1), dtype=wp.float64, device=self.device)
            self.eigenvectors = wp.empty_like(self.core.eigenvectors)
        if rayleigh:
            self.numerator = wp.zeros((self.k, 1), dtype=wp.vec2d, device=self.device)
            self.denominator = wp.zeros((self.k, 1), dtype=wp.float64, device=self.device)

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
        if self.eigenvectors.ptr != source.ptr:
            wp.copy(self.eigenvectors, source)
        super().finalize()
        if self.rayleigh:
            # For fixed x, this lambda minimizes ||Ax-lambda Bx||_2. It is
            # postprocessing, not a replacement projected eigensolve, and
            # cannot certify the requested spectrum or fix a wrong subspace.
            evaluation = self.evaluation
            self.numerator.zero_()
            self.denominator.zero_()
            wp.launch_tiled(
                quotient_parts,
                dim=(self.k, (self.op.complex_n + 255) // 256),
                inputs=[evaluation.ax, evaluation.bx, self.numerator, self.denominator],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                quotient_values,
                self.k,
                [self.numerator, self.denominator, self.eigenvalues],
                device=self.device,
            )
            self.refresh_counts()

    def refresh_counts(self):
        evaluation = self.evaluation
        evaluation.norms.zero_()
        wp.launch_tiled(
            pencil_norms,
            dim=(self.k, (self.op.complex_n + 255) // 256),
            inputs=[evaluation.ax, evaluation.bx, self.eigenvalues, evaluation.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            pencil_errors,
            self.k,
            [self.eigenvalues, evaluation.norms, evaluation.errors_squared],
            device=self.device,
        )
        self.converged.zero_()
        wp.launch(
            pencil_outputs,
            self.k,
            [evaluation.errors_squared, self.residuals, self.status, self.converged, self.tol],
            device=self.device,
        )
