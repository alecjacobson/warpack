"""Symmetry restoration must preserve original equations and captured replay."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.bounded_jacobi import BoundedJacobi
from tests.test_compact_jacobi import allocate


@pytest.mark.parametrize("n", [2, 48, 104, 128, 208, 312, 512, 612])
@pytest.mark.parametrize("kind", ["random", "rank_deficient"])
def test_original_equation_spectrum_strides_and_changed_replay(n, kind):
    rng = np.random.default_rng(159)
    a = rng.normal(size=(n, n))
    if kind == "random":
        a = (a + a.T) * 0.5  # indefinite as well as symmetric
    else:
        q, _ = la.qr(a)
        d = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
        d[: n // 3] = 0
        a = (q * d) @ q.T
    original = a.copy()
    solver, stores = allocate(n, constructor=partial(BoundedJacobi, limit=30, resymmetrize=True))
    # Precompile before capturing: subsequent replay has no lazy initialization.
    solver.h.assign(original)
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for scale in (1.0, 1e-80, 1e80):
        a = original * scale
        solver.h.assign(a)
        wp.capture_launch(capture.graph)
        values = np.diag(solver.h.numpy()) / scale
        vectors = solver.v.numpy()
        np.testing.assert_allclose(np.sort(values), la.eigvalsh(original), atol=1e-11, rtol=1e-12)
        norm = max(np.linalg.norm(original), 1)
        assert np.linalg.norm(original @ vectors - vectors * values) / norm < 1e-12
        assert np.linalg.norm(vectors.T @ vectors - np.eye(n)) < 1e-11
        assert 1 <= solver.sweeps.numpy()[0] <= 30
        h = solver.h.numpy() / scale
        np.testing.assert_array_equal(h, h.T)
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)
