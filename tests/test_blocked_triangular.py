"""Independent triangular equations, changed replay and both full QR passes."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.blocked_triangular import BlockedTriangularBlock, blocked_triangular
from tests.test_triangular_rotate import (
    test_orthogonalization_with_changed_mass_scaling_and_cached_images as check_full_stage,
)


@pytest.mark.parametrize("width", [7, 8, 9, 52, 104, 204])
def test_equations_strides_and_replay(width, columns=32):
    n = 77
    rng = np.random.default_rng(1173)
    lower = np.tril(rng.normal(size=(width, width)) * 0.05, -1) + np.eye(width)
    store = wp.full((width, n + 5), 17.0, dtype=wp.float64)
    factor_store = wp.full((width, width + 3), 19.0, dtype=wp.float64)
    q, factor = store[:, :n], factor_store[:, :width]
    factor.assign(lower)
    q.assign(rng.normal(size=(width, n)))

    def run():
        wp.launch_tiled(
            blocked_triangular,
            dim=(n + columns - 1) // columns,
            inputs=[q, factor],
            block_dim=columns,
        )

    run()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        run()
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        current = lower.copy()
        current[np.diag_indices(width)] *= 1.25 if scale == 1e80 else 1
        source = scale * rng.normal(size=(width, n))
        q.assign(source)
        factor.assign(current)
        wp.capture_launch(capture.graph)
        result = q.numpy()
        reference = la.solve_triangular(current, source, lower=True)
        np.testing.assert_allclose(result, reference, atol=abs(scale) * 1e-13, rtol=1e-11)
        bound = max(abs(scale), 1e-100)
        assert la.norm((current @ result - source) / bound) < 1e-11
        assert np.all(store.numpy()[:, n:] == 17)
        assert np.all(factor_store.numpy()[:, width:] == 19)
        np.testing.assert_array_equal(factor.numpy(), current)


@pytest.mark.parametrize("width", [8, 52, 104, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_full_two_pass_stage(width, cached):
    check_full_stage(width, cached, BlockedTriangularBlock)


@pytest.mark.parametrize("width", [7, 104, 204])
def test_unused_upper_factor_triangle_is_never_read(width, columns=32):
    n = 67
    rng = np.random.default_rng(8173)
    lower = np.tril(rng.normal(size=(width, width)) * 0.05, -1) + np.eye(width)
    poisoned = lower.copy()
    poisoned[np.triu_indices(width, 1)] = np.nan
    factor = wp.array(poisoned, dtype=wp.float64)
    source = rng.normal(size=(width, n))
    q = wp.array(source, dtype=wp.float64)
    wp.launch_tiled(
        blocked_triangular, dim=(n + columns - 1) // columns, inputs=[q, factor], block_dim=columns
    )
    result = q.numpy()
    assert np.isfinite(result).all()
    assert la.norm(lower @ result - source) / la.norm(source) < 1e-13
    np.testing.assert_array_equal(factor.numpy(), poisoned)
