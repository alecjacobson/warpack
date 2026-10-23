"""Even, dense copyout can vectorize and change which thread reads each entry."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.blocked_triangular_columns import column_kernel


@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("n", [1024, 70656])
def test_identity_solve_survives_vectorized_copyout(n, columns):
    width = 120
    source = np.random.default_rng(178).normal(size=(width, n))
    q = wp.array(source, dtype=wp.float64)
    factor = wp.array(np.eye(width), dtype=wp.float64)
    kernel = column_kernel(columns)

    def run():
        wp.launch_tiled(
            kernel, dim=(n + columns - 1) // columns, inputs=[q, factor], block_dim=columns
        )

    run()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        run()
    for scale in [1.0, -0.75, 1.25, 1.0, 0.5]:
        current = scale * source
        q.assign(current)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(q.numpy(), current)
