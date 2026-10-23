"""Gram alternatives retain original products, strided tails and changed replays."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.gram_tiles import GramTileBlock, gram
from tests import test_certified_whitening, test_first_whitening
from tests.test_triangular_rotate import (
    test_orthogonalization_with_changed_mass_scaling_and_cached_images as check_stage,
)


@pytest.mark.parametrize("tile", [8, 16, 32])
@pytest.mark.parametrize("threads", [64, 128, 256])
@pytest.mark.parametrize("symmetric", [False, True])
def test_strided_rectangular_scaled_changed_replay(tile, threads, symmetric):
    rows, columns, physical = 19, 19 if symmetric else 27, 1031
    rng = np.random.default_rng(180)
    xbase, ybase = rng.normal(size=(rows, physical)), rng.normal(size=(columns, physical))
    stores = [wp.full((n, physical + 7), 17.0, dtype=wp.float64) for n in (rows, columns)]
    x, y = [a[:, :physical] for a in stores]
    result = wp.full((rows + 3, columns + 5), -17.0, dtype=wp.float64)
    out = result[:rows, :columns]
    gram(x, y, out, tile=tile, threads=threads, symmetric=symmetric)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        gram(x, y, out, tile=tile, threads=threads, symmetric=symmetric)
    for iteration, scale in enumerate([1.0, 1e-60, 1e60, 0.0, -1.0]):
        current_x = xbase.copy()
        current_x[0] += 0.125 * iteration
        current_y = current_x * np.linspace(0.5, 2.0, physical) if symmetric else ybase
        expected = current_x @ current_y.T
        x.assign(scale * current_x)
        y.assign(scale * current_y)
        wp.capture_launch(capture.graph)
        actual = out.numpy()
        if scale:
            assert np.linalg.norm(actual / scale**2 - expected) / np.linalg.norm(expected) < 5e-13
        else:
            np.testing.assert_array_equal(actual, np.zeros_like(expected))
        assert np.all(result.numpy()[rows:] == -17) and np.all(result.numpy()[:, columns:] == -17)
        for store in stores:
            assert np.all(store.numpy()[:, physical:] == 17)


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("width", [96, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_complete_two_pass_stage(tile, width, cached):
    class Candidate(GramTileBlock):
        _gram_output_tile = tile

    check_stage(width, cached, Candidate)


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("history", [False, True])
def test_changed_coupled_constraints_clusters(monkeypatch, tmp_path, tile, history):
    class Candidate(GramTileBlock):
        _gram_output_tile = tile

    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", Candidate)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )
