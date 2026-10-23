"""Register blocking must preserve the complete serial arithmetic recurrence."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.register_triangular import RegisterTriangularBlock, register_triangular
from tests.test_triangular_rotate import (
    test_orthogonalization_with_changed_mass_scaling_and_cached_images as check_full_stage,
)
from warpack import kernels as K


@pytest.mark.parametrize("width", [7, 17, 52, 104, 120, 204, 256])
def test_exact_recurrence_strides_poisoned_upper_and_changed_replay(width):
    n = 1031
    rng = np.random.default_rng(10173)
    factor = np.tril(rng.normal(size=(width, width)) * 0.05, -1) + np.eye(width)
    poisoned = factor.copy()
    poisoned[np.triu_indices(width, 1)] = np.nan
    lower = wp.array(poisoned, dtype=wp.float64)
    backing = [wp.full((width, n + 5), 17.0, dtype=wp.float64) for _ in range(2)]
    targets = [a[:, :n] for a in backing]
    targets[1].assign(rng.normal(size=(width, n)))
    wp.launch(register_triangular, n, [targets[1], lower], block_dim=256)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.launch(register_triangular, n, [targets[1], lower], block_dim=256)
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        current_factor = factor.copy()
        current_factor[np.diag_indices(width)] *= 1.25 if scale == 1e80 else 1
        poisoned = current_factor.copy()
        poisoned[np.triu_indices(width, 1)] = np.nan
        lower.assign(poisoned)
        source = rng.normal(size=(width, n)) * scale
        for target in targets:
            target.assign(source)
        wp.launch(K.triangular, n, [targets[0], lower], block_dim=256)
        wp.capture_launch(capture.graph)
        result = targets[1].numpy()
        np.testing.assert_array_equal(targets[0].numpy(), result)
        assert la.norm((current_factor @ result - source) / max(abs(scale), 1e-100)) < 1e-10
        for store in backing:
            assert np.all(store.numpy()[:, n:] == 17)
        np.testing.assert_array_equal(lower.numpy(), poisoned)


def test_large_even_identity_replay():
    n, width = 70656, 120
    source = np.random.default_rng(178).normal(size=(width, n))
    q, factor = wp.array(source, dtype=wp.float64), wp.array(np.eye(width), dtype=wp.float64)
    wp.launch(register_triangular, n, [q, factor], block_dim=256)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.launch(register_triangular, n, [q, factor], block_dim=256)
    for scale in [1.0, -0.75, 1.25]:
        current = scale * source
        q.assign(current)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(q.numpy(), current)


@pytest.mark.parametrize("width", [52, 120, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_full_two_pass_stage(width, cached):
    check_full_stage(width, cached, RegisterTriangularBlock)
