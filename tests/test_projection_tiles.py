"""Projection updates must preserve strided input views and both projection passes."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental import projection_tiles as candidate
from warpack.block_products import subtract_projection


@pytest.mark.parametrize("rows", [None, 4, 8, 16])
@pytest.mark.parametrize("width,basis,n", [(7, 3, 37), (13, 0, 63), (48, 24, 257), (8, 204, 1031)])
def test_strided_partial_changed_replay_and_padding(rows, width, basis, n):
    rng = np.random.default_rng(192)
    qstore = wp.full((basis + 1, n + 3), 17.0, dtype=wp.float64)
    cstore = wp.full((width + 1, basis + 5), 17.0, dtype=wp.float64)
    ostore = wp.full((width + 1, n + 7), -17.0, dtype=wp.float64)
    q, coeff, out = qstore[:basis, :n], cstore[:width, :basis], ostore[:width, :n]

    def run():
        if rows is None:
            wp.launch_tiled(
                subtract_projection,
                dim=((width + 3) // 4, (n + 31) // 32),
                inputs=[q, coeff, out],
                block_dim=128,
            )
        else:
            candidate.subtract(q, coeff, out, rows=rows)

    run()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        run()
    for scale in (1.0, 1e100, 1e-100, 0.0, 1.0):
        hq = rng.normal(size=q.shape) * scale
        hc = rng.normal(size=coeff.shape) * scale
        initial = rng.normal(size=out.shape) * scale * scale
        for dest, values in [(q, hq), (coeff, hc), (out, initial)]:
            if values.size:
                dest.assign(values)
        expected = initial - hc @ hq
        wp.capture_launch(capture.graph)
        actual = out.numpy()
        largest = np.max(abs(expected))
        error = np.max(abs(actual - expected))
        assert np.isfinite(actual).all()
        assert error <= 5e-13 * largest if largest else error == 0
        hb, hcback, ho = qstore.numpy(), cstore.numpy(), ostore.numpy()
        np.testing.assert_array_equal(hb[:basis, :n], hq)
        np.testing.assert_array_equal(hcback[:width, :basis], hc)
        assert np.all(hb[basis:] == 17) and np.all(hb[:, n:] == 17)
        assert np.all(hcback[width:] == 17) and np.all(hcback[:, basis:] == 17)
        assert np.all(ho[width:] == -17) and np.all(ho[:, n:] == -17)


@pytest.mark.parametrize("rows", [4, 8, 16])
@pytest.mark.parametrize("history", [False, True])
def test_coupled_metric_constraints_clusters_changed_problem(monkeypatch, tmp_path, rows, history):
    from tests import test_certified_whitening, test_first_whitening

    class Candidate(candidate.ProjectionBlock):
        _projection_rows = rows

    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", Candidate)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )
