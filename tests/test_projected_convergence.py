"""Projected FP64 accuracy over spectra that exceed the old ten-sweep budget."""

import re

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator, SymmetricEigensolver


@pytest.mark.parametrize("n", [48, 64, 72, 96, 128])
@pytest.mark.parametrize("indefinite", [False, True])
@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_projected_complete_spectrum_scaled_replay(n, indefinite, scale, tmp_path):
    rng = np.random.default_rng(831)
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    values = np.geomspace(400, 0.0025, n)
    if indefinite:
        values[3::2] *= -1
    values[:3] = 400
    values[-2:] = 0
    original = (q * (values * scale)) @ q.T
    solver = SymmetricEigensolver(SparseOperator(matrix(original)), 20, ncv=n, which="LA")
    source = wp.array(original, dtype=wp.float64)
    wp.copy(solver.g, source)
    solver.diagonalize()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.copy(solver.g, source)
        solver.diagonalize()
    for factor, shift in ((1.0, 0.0), (2.0, -0.25), (1.0, 0.0)):
        expected = scale * (factor * values + shift)
        host = (q * expected) @ q.T
        source.assign(host)
        wp.capture_launch(capture.graph)
        v, w = solver.v.numpy(), solver.ritz.numpy()
        denominator = max(abs(expected))
        assert np.linalg.norm(host @ v - v * w, axis=0).max() / denominator < 1e-11
        assert abs(np.sort(w) - np.sort(expected)).max() / denominator < 1e-11
        np.testing.assert_allclose(v.T @ v, np.eye(n), atol=1e-11, rtol=0)
        # Complete leading repeated and near-zero spaces, independently known.
        for ids in (np.arange(3), np.arange(n - 2, n)):
            target = expected[ids[0]]
            chosen = np.argsort(abs(w - target))[: len(ids)]
            basis = q[:, ids]
            assert np.linalg.norm(v[:, chosen] - basis @ (basis.T @ v[:, chosen])) < 1e-8
        np.testing.assert_array_equal(source.numpy(), host)
    dot = tmp_path / "projected.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("captured", [False, True])
def test_projected_runtime_without_conditionals(monkeypatch, captured):
    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: False)
    n = 72
    rng = np.random.default_rng(831)
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    values = np.geomspace(400, 0.0025, n)
    values[:3] = 400
    values[-2:] = 0
    host = (q * values) @ q.T
    solver = SymmetricEigensolver(SparseOperator(matrix(host)), 20, ncv=n, which="LA")
    assert solver._global_jacobi is None
    source = wp.array(host, dtype=wp.float64)
    wp.copy(solver.g, source)
    solver.diagonalize()
    wp.synchronize()
    if captured:
        with wp.ScopedCapture() as capture:
            wp.copy(solver.g, source)
            solver.diagonalize()
        wp.capture_launch(capture.graph)
    v, w = solver.v.numpy(), solver.ritz.numpy()
    assert np.linalg.norm(host @ v - v * w, axis=0).max() / 400 < 1e-11
    assert abs(np.sort(w) - np.sort(values)).max() / 400 < 1e-11
    np.testing.assert_allclose(v.T @ v, np.eye(n), atol=1e-11, rtol=0)
