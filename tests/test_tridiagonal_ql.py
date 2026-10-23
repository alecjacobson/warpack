"""Independent tridiagonal spectrum, vectors, status and replay coverage."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.tridiagonal_ql import TridiagonalQL
from benchmarks.experimental.tridiagonal_ql_rows import RowTridiagonalQL


@pytest.mark.parametrize("constructor", [TridiagonalQL, RowTridiagonalQL])
@pytest.mark.parametrize("n", [1, 2, 8, 48, 104, 208, 312, 512])
@pytest.mark.parametrize("kind", ["indefinite", "clustered", "repeated"])
def test_tridiagonal_original_equation_and_replay(n, kind, constructor):
    rng = np.random.default_rng(161)
    d, e = rng.normal(size=n), rng.normal(size=max(0, n - 1))
    if kind == "clustered":
        d = np.ones(n) + d * 1e-12
        e *= 1e-12
    elif kind == "repeated":
        d = np.arange(n, dtype=float) // 4
        e *= 0
    original = np.diag(d) + np.diag(e, 1) + np.diag(e, -1) if n > 1 else np.diag(d)
    expected = la.eigvalsh(original)
    # Both inputs are strided views.
    dd, ee = (
        wp.zeros(2 * n, dtype=wp.float64)[::2],
        (wp.zeros(2 * (n - 1), dtype=wp.float64)[::2] if n > 1 else wp.empty(0, dtype=wp.float64)),
    )
    solver = constructor(dd, ee)
    dd.assign(d)
    ee.assign(e)
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for scale in [1.0, 1e-80, 1e80]:
        dd.assign(d * scale)
        ee.assign(e * scale)
        wp.capture_launch(capture.graph)
        assert np.all(solver.state.numpy()[:, 1] == 0)
        values, vectors = solver.values.numpy() / scale, solver.vectors.numpy()
        norm = max(1, np.linalg.norm(original))
        np.testing.assert_allclose(np.sort(values), expected, atol=1e-13 * norm, rtol=1e-12)
        assert np.linalg.norm(original @ vectors - vectors * values) / norm < 1e-12
        assert np.linalg.norm(vectors.T @ vectors - np.eye(n)) < 1e-11


@pytest.mark.parametrize("constructor", [TridiagonalQL, RowTridiagonalQL])
def test_budget_nonfinite_and_recovery(constructor):
    d, e = wp.ones(8, dtype=wp.float64), wp.ones(7, dtype=wp.float64)
    exhausted = constructor(d, e, maxiter=0)
    exhausted.apply()
    assert np.all(exhausted.state.numpy()[:, 1] == 2)
    solver = constructor(d, e)
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for value, status in [(np.nan, 1), (0.0, 0), (1.0, 0)]:
        d.fill_(value)
        e.fill_(value)
        wp.capture_launch(capture.graph)
        assert np.all(solver.state.numpy()[:, 1] == status)
        if value == 0.0:
            np.testing.assert_array_equal(solver.values.numpy(), np.zeros(8))
            np.testing.assert_array_equal(solver.vectors.numpy(), np.eye(8))
