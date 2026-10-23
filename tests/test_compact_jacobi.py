"""Compact-loop projected work must retain original equations and replay safety."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.compact_jacobi import CompactJacobi
from warpack.projected import GlobalJacobi


def allocate(n, padding=3, constructor=CompactJacobi):
    stores = [wp.full((n, n + padding), 17.0, dtype=wp.float64) for _ in range(4)]
    h, v, temp, vtemp = (a[:, :n] for a in stores)
    cs, ss = (wp.empty(n, dtype=wp.float64) for _ in range(2))
    partners = wp.empty(n, dtype=wp.int32)
    return constructor(h, v, cs, ss, partners, temp, vtemp), stores


@pytest.mark.parametrize("n", [2, 48, 104, 128, 208, 312, 512, 612])
@pytest.mark.parametrize("kind", ["random", "clustered"])
@pytest.mark.parametrize("rounds", [2, 16, 32])
def test_original_equation_spectrum_strides_and_changed_replay(n, kind, rounds):
    rng = np.random.default_rng(159)
    a = rng.normal(size=(n, n))
    if kind == "random":
        a = (a + a.T) * 0.5  # indefinite as well as symmetric
    else:
        q, _ = la.qr(a)
        a = (q * np.repeat(np.arange((n + 3) // 4), 4)[:n]) @ q.T
    original = a.copy()
    solver, stores = allocate(n, constructor=partial(CompactJacobi, rounds_per_loop=rounds))
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
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)


@pytest.mark.parametrize("rounds", [2, 16, 32])
def test_zero_nonfinite_and_recovery(rounds):
    solver, _ = allocate(8, constructor=partial(CompactJacobi, rounds_per_loop=rounds))
    for source, expected in [(np.zeros((8, 8)), 1), (np.full((8, 8), np.nan), 30), (np.eye(8), 1)]:
        solver.h.assign(source)
        solver.apply()
        assert solver.sweeps.numpy()[0] == expected
        if expected == 1:
            np.testing.assert_array_equal(solver.h.numpy(), source)
            np.testing.assert_array_equal(solver.v.numpy(), np.eye(8))


@pytest.mark.parametrize("n", [1, 3, 1026])
def test_reject_unsupported_dimensions(n):
    with pytest.raises(ValueError, match="even dimension"):
        allocate(n)


@pytest.mark.parametrize("n", [2, 48, 104, 208, 612])
@pytest.mark.parametrize("rounds", [2, 8, 16, 32])
def test_preserves_installed_round_order(n, rounds):
    rng = np.random.default_rng(164)
    source = rng.normal(size=(n, n))
    source = (source + source.T) * 0.5
    outputs = []
    for constructor in (GlobalJacobi, partial(CompactJacobi, rounds_per_loop=rounds)):
        solver, _ = allocate(n, constructor=constructor)
        solver.h.assign(source)
        solver.apply()
        outputs.append((solver.h.numpy(), solver.v.numpy(), solver.sweeps.numpy()))
    # Same arithmetic and rotation ordering, including orientation of vectors;
    # an equivalent sorted spectrum alone would not prove the same round order.
    for baseline, candidate in zip(*outputs):
        np.testing.assert_array_equal(candidate, baseline)
