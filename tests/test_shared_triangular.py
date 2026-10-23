"""Shared recurrence must retain the original equations and subtraction order."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.shared_triangular import SharedTriangularBlock, shared_kernel
from tests import test_blocked_triangular as original
from warpack import kernels as K


@pytest.mark.parametrize("width", [7, 17, 104, 120, 204, 256])
def test_shared_equations_strides_and_replay(monkeypatch, width):
    monkeypatch.setattr(original, "blocked_triangular", shared_kernel(width))
    original.test_equations_strides_and_replay(width)


@pytest.mark.parametrize("width", [104, 120, 204])
def test_bitwise_serial_recurrence(width):
    n = 1031
    rng = np.random.default_rng(9173)
    factor = np.tril(rng.normal(size=(width, width)) * 0.05, -1) + np.eye(width)
    lower = wp.array(factor, dtype=wp.float64)
    targets = [wp.empty((width, n), dtype=wp.float64) for _ in range(2)]
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        source = rng.normal(size=(width, n)) * scale
        for target in targets:
            target.assign(source)
        wp.launch(K.triangular, n, [targets[0], lower], block_dim=256)
        wp.launch_tiled(
            shared_kernel(width), dim=(n + 31) // 32, inputs=[targets[1], lower], block_dim=32
        )
        np.testing.assert_array_equal(targets[0].numpy(), targets[1].numpy())


@pytest.mark.parametrize("width", [7, 104, 204])
def test_shared_unused_upper_triangle(monkeypatch, width):
    monkeypatch.setattr(original, "blocked_triangular", shared_kernel(width))
    original.test_unused_upper_factor_triangle_is_never_read(width)


@pytest.mark.parametrize("width", [52, 120, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_shared_full_stage(width, cached):
    original.check_full_stage(width, cached, SharedTriangularBlock)
