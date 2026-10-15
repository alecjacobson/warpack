"""Relative residual preconditioning must not silently discard small RHSs."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CGInverse


def test_relative_only_cg_replay_across_rhs_scales():
    diagonal = np.linspace(1, 2, 32)
    inverse = CGInverse(matrix(sp.diags(diagonal, format="csr")), tol=0.01, atol=0.0, maxiter=10)
    b, x = wp.ones((2, 32), dtype=wp.float64), wp.empty((2, 32), dtype=wp.float64)
    inverse.apply(b, x)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        inverse.reset_stats()
        inverse.apply(b, x)
    for scale in [1.0, 1e-8, 1e-20, 1e8]:
        source = np.full((2, 32), scale)
        b.assign(source)
        wp.capture_launch(capture.graph)
        actual = x.numpy()
        residual = np.linalg.norm(actual * diagonal - source, axis=1)
        assert np.max(residual / np.linalg.norm(source, axis=1)) < 0.01
        iterations, maximum, unconverged, count = inverse.stats.numpy()
        assert 0 < maximum <= 10 and iterations > 0 and unconverged == 0 and count == 2


def test_default_keeps_upstream_absolute_tolerance_and_zero_rhs():
    inverse = CGInverse(matrix(sp.eye(16, format="csr")), tol=0.01)
    b = wp.full((1, 16), 1e-8, dtype=wp.float64)
    x = wp.empty_like(b)
    inverse.apply(b, x)
    np.testing.assert_array_equal(x.numpy(), 0)
    assert inverse.stats.numpy()[0] == 0
    relative = CGInverse(matrix(sp.eye(16, format="csr")), tol=0.01, atol=0.0)
    b.zero_()
    relative.apply(b, x)
    np.testing.assert_array_equal(x.numpy(), 0)
    assert relative.stats.numpy()[2] == 0


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf")])
def test_invalid_absolute_tolerance(atol):
    with pytest.raises(ValueError, match="atol"):
        CGInverse(matrix(sp.eye(4, format="csr")), atol=atol)
