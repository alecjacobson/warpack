"""Installed projected solves must converge on repeated/null spectra (#170)."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from tests.test_compact_jacobi import allocate
from warpack._compact_jacobi import CompactJacobi
from warpack.projected import GlobalJacobi


@pytest.mark.parametrize("n", [2, 48, 104, 128, 208, 312, 512, 612])
@pytest.mark.parametrize("kind", ["random", "rank_deficient"])
@pytest.mark.parametrize("constructor", [GlobalJacobi, CompactJacobi])
def test_original_equation_spectrum_strides_and_changed_replay(n, kind, constructor):
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
    solver, stores = allocate(n, constructor=constructor)
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


@pytest.mark.parametrize("constructor", [GlobalJacobi, CompactJacobi])
def test_repeated_nullspace_original_equation_changed_replay(constructor):
    n = 612
    q, _ = la.qr(np.random.default_rng(159).normal(size=(n, n)))
    d = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
    d[: n // 3] = 0
    a = (q * d) @ q.T
    limit = 30
    solver, stores = allocate(n, constructor=constructor)
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
        h = solver.h.numpy() / scale
        assert np.max(abs(h - np.diag(values))) <= 2e-15 * np.max(abs(values))
        np.testing.assert_array_equal(h, h.T)
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)
