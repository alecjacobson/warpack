"""Experimental launch fusion preserves each dense rotation and complete spectra."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.jacobi_on_the_fly import OnTheFlyJacobi, scalar_round, tiled_round
from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator, SymmetricEigensolver


@pytest.mark.parametrize("n", [2, 6, 16, 18, 64, 104, 208, 312])
@pytest.mark.parametrize("variant", ["scalar", "tile16", "tile32"])
def test_one_round_matches_independent_plane_rotations(n, variant):
    rng = np.random.default_rng(154)
    raw = rng.normal(size=(n, n))
    h = raw + raw.T + np.diag(np.linspace(-20, 30, n))
    v = rng.normal(size=(n, n))
    wh, wv = wp.array(h, dtype=wp.float64), wp.array(v, dtype=wp.float64)
    oh, ov = wp.empty_like(wh), wp.empty_like(wv)
    for r in sorted({0, n // 2 - 1, n - 2}):
        j = np.eye(n)
        for i in range(n // 2):
            p, q = (r + i) % (n - 1), (r - i + n - 1) % (n - 1)
            if i == 0:
                q = n - 1
            c, s = 1.0, 0.0
            if abs(h[p, q]) > 1e-16 * (abs(h[p, p]) + abs(h[q, q])):
                tau = (h[q, q] - h[p, p]) / (2 * h[p, q])
                t = np.copysign(1 / (abs(tau) + np.sqrt(1 + tau * tau)), tau)
                c = 1 / np.sqrt(1 + t * t)
                s = t * c
            j[p, p] = j[q, q] = c
            j[p, q], j[q, p] = s, -s
        if variant == "scalar":
            wp.launch(scalar_round, (n, n), [wh, wv, oh, ov, r])
        else:
            size = int(variant[4:])
            wp.launch_tiled(
                tiled_round(size),
                dim=((n + size - 1) // size,) * 2,
                inputs=[wh, wv, oh, ov, r],
                block_dim=256,
            )
        np.testing.assert_allclose(oh.numpy(), j.T @ h @ j, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(ov.numpy(), v @ j, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("n", [48, 104, 208])
@pytest.mark.parametrize("variant", ["scalar", "tile16", "tile32"])
def test_complete_spectrum_and_changed_input_replay(n, variant):
    rng = np.random.default_rng(1541)
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    spectrum = np.geomspace(400, 0.0025, n)
    spectrum[:3], spectrum[-3:] = 400, [-1, -1, 0]
    original = (q * spectrum) @ q.T
    solver = SymmetricEigensolver(SparseOperator(matrix(original)), 20, ncv=n, which="LA")
    solver._global_jacobi = OnTheFlyJacobi(
        solver.g,
        solver.v,
        solver.cs,
        solver.ss,
        solver.partners,
        solver.jtmp,
        solver.vtmp,
        variant=variant,
    )
    source = wp.array(original, dtype=wp.float64)
    wp.copy(solver.g, source)
    solver.diagonalize()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.copy(solver.g, source)
        solver.diagonalize()
    for scale in (3.0, 1.0):
        source.assign(original * scale)
        wp.capture_launch(capture.graph)
        values, vectors = solver.ritz.numpy(), solver.v.numpy()
        expected = original * scale
        assert (
            max(np.linalg.norm(expected @ vectors - vectors * values, axis=0)) / (400 * scale)
            < 1e-11
        )
        assert np.max(abs(vectors.T @ vectors - np.eye(n))) < 1e-11
        np.testing.assert_allclose(
            np.sort(values), np.sort(spectrum * scale), rtol=1e-9, atol=1e-10
        )
        assert 1 <= solver._global_jacobi.sweeps.numpy()[0] <= 30
