"""Independent equation and replay checks for the block Jacobi experiment."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.block_jacobi import BlockJacobiPolish
from tests.test_compact_jacobi import allocate


@pytest.mark.parametrize("n", [9, 48, 104, 208, 312, 416, 612])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("kind", ["indefinite", "rank_deficient"])
def test_original_equation_spectrum_padding_and_changed_input(n, block_size, kind):
    rng = np.random.default_rng(169)
    a = rng.normal(size=(n, n))
    if kind == "indefinite":
        original = (a + a.T) * 0.5
    else:
        q, _ = la.qr(a)
        diagonal = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
        diagonal[: n // 3] = 0
        original = (q * diagonal) @ q.T
    solver, stores = allocate(n, constructor=partial(BlockJacobiPolish, block_size=block_size))
    solver.h.assign(original)
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for shift, scale in [(0.0, 1.0), (0.0, 1e-80), (0.5, 1e80), (-0.25, 1.0)]:
        matrix = original + shift * np.eye(n)
        solver.h.assign(matrix * scale)
        wp.capture_launch(capture.graph)
        values = np.diag(solver.h.numpy()) / scale
        vectors = solver.v.numpy()
        np.testing.assert_allclose(np.sort(values), la.eigvalsh(matrix), atol=1e-10, rtol=1e-11)
        assert (
            np.linalg.norm(matrix @ vectors - vectors * values) / max(np.linalg.norm(matrix), 1)
            < 1e-11
        )
        assert np.linalg.norm(vectors.T @ vectors - np.eye(n)) < 1e-11
        assert 1 <= solver.sweeps.numpy()[0] <= 30
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_zero_nonfinite_and_recovery(block_size):
    solver, _ = allocate(9, constructor=partial(BlockJacobiPolish, block_size=block_size))
    solver.h.zero_()
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for source, sweeps in [(np.zeros((9, 9)), 1), (np.full((9, 9), np.nan), 30), (np.eye(9), 1)]:
        solver.h.assign(source)
        wp.capture_launch(capture.graph)
        assert solver.sweeps.numpy()[0] == sweeps
        if sweeps == 1:
            np.testing.assert_array_equal(solver.h.numpy(), source)
            np.testing.assert_array_equal(solver.v.numpy(), np.eye(9))


@pytest.mark.parametrize("n", [104, 416, 612])
@pytest.mark.parametrize("initializers", [1, 4])
def test_alternate_initializer_budgets(n, initializers):
    rng = np.random.default_rng(169)
    q, _ = la.qr(rng.normal(size=(n, n)))
    values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
    values[: n // 3] = 0
    original = (q * values) @ q.T
    solver, _ = allocate(n, constructor=partial(BlockJacobiPolish, initializer_sweeps=initializers))
    solver.h.assign(original)
    solver.apply()
    v, actual = solver.v.numpy(), np.diag(solver.h.numpy())
    np.testing.assert_allclose(np.sort(actual), la.eigvalsh(original), atol=1e-10, rtol=1e-11)
    assert la.norm(original @ v - v * actual) / la.norm(original) < 1e-11
    assert la.norm(v.T @ v - np.eye(n)) < 1e-11


@pytest.mark.parametrize("n", [1023, 1024])
def test_largest_padded_dimensions(n):
    rng = np.random.default_rng(169)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    solver, _ = allocate(n, constructor=BlockJacobiPolish)
    solver.h.assign(a)
    solver.apply()
    v, values = solver.v.numpy(), np.diag(solver.h.numpy())
    np.testing.assert_allclose(np.sort(values), la.eigvalsh(a), atol=1e-10, rtol=1e-11)
    assert la.norm(a @ v - v * values) / la.norm(a) < 1e-11
    assert la.norm(v.T @ v - np.eye(n)) < 1e-11
