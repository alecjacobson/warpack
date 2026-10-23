"""A recorded projected sweep-limit miss needs bounded, replay-safe recovery."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.bounded_jacobi import BoundedJacobi
from tests.test_compact_jacobi import allocate


@pytest.mark.parametrize("symmetry", [False, True])
def test_repeated_nullspace_original_equation_changed_replay(symmetry):
    n = 612
    q, _ = la.qr(np.random.default_rng(159).normal(size=(n, n)))
    d = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
    d[: n // 3] = 0
    a = (q * d) @ q.T
    limit = 30 if symmetry else 60
    solver, stores = allocate(
        n, constructor=partial(BoundedJacobi, limit=limit, resymmetrize=symmetry)
    )
    solver.h.assign(a)
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for scale, shift in [(1.0, 0.0), (1e-80, 0.0), (1e80, 0.0), (1.0, 0.5)]:
        source = a + shift * np.eye(n)
        solver.h.assign(source * scale)
        wp.capture_launch(capture.graph)
        v, values = solver.v.numpy(), np.diag(solver.h.numpy()) / scale
        np.testing.assert_allclose(np.sort(values), d + shift, atol=1e-10, rtol=1e-11)
        assert la.norm(source @ v - v * values) / la.norm(source) < 1e-11
        assert la.norm(v.T @ v - np.eye(n)) < 1e-11
        assert 1 <= solver.sweeps.numpy()[0] <= limit
        if symmetry:
            h = solver.h.numpy() / scale
            assert np.max(abs(h - np.diag(values))) <= 2e-15 * np.max(abs(values))
            np.testing.assert_array_equal(h, h.T)
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)


@pytest.mark.parametrize("limit", [30, 60])
def test_nonfinite_is_bounded_and_next_replay_recovers(limit):
    solver, _ = allocate(8, constructor=partial(BoundedJacobi, limit=limit))
    solver.h.assign(np.eye(8))
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for value, expected in [(np.nan, limit), (1.0, 1)]:
        source = np.full((8, 8), value) if np.isnan(value) else np.eye(8)
        solver.h.assign(source)
        wp.capture_launch(capture.graph)
        assert solver.sweeps.numpy()[0] == expected
        if value == 1.0:
            np.testing.assert_array_equal(solver.h.numpy(), source)
            np.testing.assert_array_equal(solver.v.numpy(), source)


@pytest.mark.parametrize("n", [48, 104, 208, 612])
def test_early_convergence_preserves_original_arithmetic(n):
    a = np.random.default_rng(170).normal(size=(n, n))
    a = (a + a.T) * 0.5
    results = []
    for constructor in [partial(BoundedJacobi, limit=30), BoundedJacobi]:
        solver, _ = allocate(n, constructor=constructor)
        solver.h.assign(a)
        solver.apply()
        results.append((solver.h.numpy(), solver.v.numpy(), solver.sweeps.numpy()))
    assert results[0][2][0] < 30
    for before, after in zip(*results):
        np.testing.assert_array_equal(after, before)
