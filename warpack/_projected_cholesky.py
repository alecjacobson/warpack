"""Cooperative projected FP64 Cholesky for wide block orthogonalization.

Keep each entry's serial subtraction order and pivot containment. This internal
kernel is used only when one CUDA block can hold the packed lower triangle.
"""

import functools

import warp as wp


@functools.lru_cache(None)
def packed_cholesky_kernel(size):
    if not 1 <= size <= 128:
        raise ValueError("cooperative factor supports dimensions 1..128")
    n = wp.constant(size)
    count = wp.constant(size * (size + 1) // 2)

    @wp.kernel(enable_backward=False, module="unique")
    def compute(
        g: wp.array2d[wp.float64], factor: wp.array2d[wp.float64], status: wp.array[wp.int32]
    ):
        block, lane = wp.tid()
        packed = wp.tile_zeros(shape=count, dtype=wp.float64, storage="shared")
        diagonal = wp.tile_zeros(shape=1, dtype=wp.float64, storage="shared")
        for j in range(n):
            i = j + lane
            value = wp.float64(0.0)
            if i < n:
                value = g[i, j]
                for k in range(j):
                    value -= wp.tile_extract(packed, i * (i + 1) // 2 + k) * wp.tile_extract(
                        packed, j * (j + 1) // 2 + k
                    )
            root = wp.float64(0.0)
            if lane == 0:
                if not wp.isfinite(value) or value <= wp.float64(0.0):
                    wp.atomic_max(status, 0, 1)
                root = wp.sqrt(wp.max(value, wp.float64(1.0e-30)))
            wp.tile_scatter_masked(diagonal, 0, root, lane == 0)
            root = wp.tile_extract(diagonal, 0)
            entry = value / root
            if lane == 0:
                entry = root
            wp.tile_scatter_masked(packed, wp.min(i * (i + 1) // 2 + j, count - 1), entry, i < n)
            if i < n:
                factor[i, j] = entry

    return compute
