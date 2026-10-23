"""Changing thread grouping must preserve triangular solve arithmetic exactly."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from warpack import kernels as K


@pytest.mark.parametrize("width", [24, 104, 204])
@pytest.mark.parametrize("block_dim", [32, 64, 128, 512])
def test_exact_triangular_launch_size_with_strides_and_replay(width, block_dim):
    n = 1031
    rng = np.random.default_rng(173)
    random = rng.normal(size=(width, width))
    factor = la.cholesky(random @ random.T + width * np.eye(width), lower=True)
    lower = wp.array(factor, dtype=wp.float64)
    backing = [wp.full((width, n + 5), 17.0, dtype=wp.float64) for _ in range(2)]
    targets = [a[:, :n] for a in backing]
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        source = rng.normal(size=(width, n)) * scale
        for target, threads in zip(targets, [256, block_dim]):
            target.assign(source)
            wp.launch(K.triangular, n, [target, lower], block_dim=threads)
        first, second = [target.numpy() for target in targets]
        np.testing.assert_array_equal(first, second)
        if scale:
            assert (
                la.norm(factor @ (second / scale) - source / scale) / la.norm(source / scale)
                < 1e-13
            )
        for store in backing:
            assert np.all(store.numpy()[:, n:] == 17.0)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.launch(K.triangular, n, [targets[1], lower], block_dim=block_dim)
    for _ in range(2):
        source = rng.normal(size=(width, n))
        for target in targets:
            target.assign(source)
        wp.launch(K.triangular, n, [targets[0], lower], block_dim=256)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(targets[0].numpy(), targets[1].numpy())
