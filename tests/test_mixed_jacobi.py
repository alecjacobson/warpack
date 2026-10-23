"""A lower-precision initializer cannot decide final projected rank or values."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.mixed_jacobi import MixedJacobi


def allocate(n):
    stores = [wp.full((n, n + 3), 17.0, dtype=wp.float64) for _ in range(4)]
    h, v, temp, vtemp = (a[:, :n] for a in stores)
    cs, ss = (wp.empty(n, dtype=wp.float64) for _ in range(2))
    partners = wp.empty(n, dtype=wp.int32)
    return MixedJacobi(h, v, cs, ss, partners, temp, vtemp), stores


@pytest.mark.parametrize("n", [2, 48, 72, 104, 128, 208])
@pytest.mark.parametrize("kind", ["indefinite", "repeated", "rank_threshold"])
def test_fp64_equation_spectrum_rank_and_changed_strided_replay(n, kind):
    rng = np.random.default_rng(160)
    random = rng.normal(size=(n, n))
    q, _ = la.qr(random)
    if kind == "indefinite":
        spectrum = np.linspace(-1, 1, n)
    elif kind == "repeated":
        spectrum = np.repeat(np.linspace(0, 1, (n + 3) // 4), 4)[:n]
    else:
        spectrum = np.linspace(0.1, 1, n)
        small = [0, 0.5e-11, 0.9e-11, 1.1e-11, 1.5e-11, 1e-8]
        spectrum[: min(n - 1, len(small))] = small[: min(n - 1, len(small))]
    original = (q * spectrum) @ q.T
    solver, stores = allocate(n)
    solver.h.assign(original)
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for scale in (1.0, 1e-80, 1e80):
        solver.h.assign(original * scale)
        wp.capture_launch(capture.graph)
        values, basis = np.diag(solver.h.numpy()) / scale, solver.v.numpy()
        assert solver.status.numpy()[0] == 0
        assert 1 <= solver.sweeps.numpy()[0] <= 30
        np.testing.assert_allclose(np.sort(values), spectrum, atol=1e-13, rtol=1e-12)
        norm = max(np.linalg.norm(original), 1)
        assert np.linalg.norm(original @ basis - basis * values) / norm < 1e-12
        assert np.linalg.norm(basis.T @ basis - np.eye(n)) < 1e-11
        if kind == "rank_threshold":
            assert np.count_nonzero(values > 1e-11 * values.max()) == np.count_nonzero(
                spectrum > 1e-11 * spectrum.max()
            )
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17)
        coarse = solver.coarse.h.numpy()
        off = coarse - np.diag(np.diag(coarse))
        assert np.max(abs(off)) <= 1e-6 * max(np.max(abs(np.diag(coarse))), 1)


def test_zero_nonfinite_status_and_graph_recovery():
    solver, _ = allocate(8)
    solver.h.assign(np.eye(8))
    solver.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.apply()
    for source, status in [(np.zeros((8, 8)), 0), (np.full((8, 8), np.nan), 1), (np.eye(8), 0)]:
        solver.h.assign(source)
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == status
        if status == 0:
            np.testing.assert_array_equal(solver.h.numpy(), source)
            np.testing.assert_array_equal(solver.v.numpy(), np.eye(8))
