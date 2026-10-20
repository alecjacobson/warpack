"""Exact state-coordinate similarity with device-side physical output checks.

Benchmark adapter for #111. The caller supplies original and balanced sparse
matrices; iteration uses balanced coordinates and returns normalized original
right modes. This is not a generalized-pencil or left-eigenvector API.
"""

import warp as wp

from warpack.hermitian import cmul, norm2
from warpack.shift import ShiftInvertEigensolver, unshift


@wp.kernel
def unscale(source: wp.array2d[wp.vec2d], result: wp.array2d[wp.vec2d], scale: wp.float64):
    i, j = wp.tid()
    factor = wp.float64(1.0)
    if j < source.shape[1] // 2:
        factor = scale
    result[i, j] = source[i, j] / factor


@wp.kernel
def row_norms(vectors: wp.array2d[wp.vec2d], norms: wp.array2d[wp.float64]):
    i, block = wp.tid()
    values = wp.tile_load(vectors[i], shape=256, offset=block * 256)
    wp.tile_atomic_add(norms[i], wp.tile_sum(wp.tile_map(norm2, values)))


@wp.kernel
def normalize(vectors: wp.array2d[wp.vec2d], norms: wp.array2d[wp.float64]):
    i, j = wp.tid()
    length = wp.sqrt(wp.max(norms[i, 0], wp.float64(1e-300)))
    vectors[i, j] = vectors[i, j] / length


@wp.kernel
def recover_velocity(vectors: wp.array2d[wp.vec2d], values: wp.array[wp.vec2d]):
    i, j = wp.tid()
    vectors[i, j + vectors.shape[1] // 2] = cmul(values[i], vectors[i, j])


class DampedCoordinates(ShiftInvertEigensolver):
    def __init__(
        self, original, balanced, *args, coordinate_scale, polish_steps=0, kinematic=False, **kwargs
    ):
        # This adapter has its own coordinate/polish/kinematic extraction.
        # Preserve its established policy until a matching candidate guard exists.
        kwargs.setdefault("preserve_converged", False)
        super().__init__(balanced, *args, **kwargs)
        self.coordinate_scale = wp.float64(coordinate_scale)
        self.polish_steps = polish_steps
        self.kinematic = kinematic
        self.polished = [wp.empty_like(self.core.eigenvectors) for _ in range(min(2, polish_steps))]
        self.op = original
        self.eigenvectors = wp.empty_like(self.core.eigenvectors)
        self.physical_norms = wp.zeros((self.k, 1), dtype=wp.float64, device=self.device)

    def normalize_physical(self, vectors):
        self.physical_norms.zero_()
        wp.launch_tiled(
            row_norms,
            dim=(self.k, (self.op.complex_n + 255) // 256),
            inputs=[vectors, self.physical_norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(normalize, vectors.shape, [vectors, self.physical_norms], device=self.device)

    def finalize(self):
        source = self.core.eigenvectors
        for step in range(self.polish_steps):
            result = self.polished[step % len(self.polished)]
            self.inverse.apply_complex(source, result)
            self.normalize_physical(result)
            source = result
        wp.launch(
            unscale,
            self.eigenvectors.shape,
            [source, self.eigenvectors, self.coordinate_scale],
            device=self.device,
        )
        if self.kinematic:
            wp.launch(
                unshift,
                self.k,
                [self.core.eigenvalues, self.value_buffer, self.inverse.sigma],
                device=self.device,
            )
            wp.launch(
                recover_velocity,
                (self.k, self.op.complex_n // 2),
                [self.eigenvectors, self.value_buffer],
                device=self.device,
            )
        self.normalize_physical(self.eigenvectors)
        super().finalize()
