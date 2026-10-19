"""Absolute accuracy and normalized stopping are separate symmetric contracts."""

import re

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import EigenpairEvaluation, KrylovSchur, SparseOperator


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("which", ["LA", "SA"])
def test_diagonal_dominant_large_eigenvalues(sparse, which, tmp_path):
    # Davidson-like matrices with eigenvalues near 1,000: a tiny normalized
    # residual need not meet an absolute 1e-10 bound. Both are checked here.
    rng = np.random.default_rng(123)
    n, k = 160, 10
    lower = np.tril(0.03 * rng.uniform(-1, 1, size=(n, n)), -1)
    if sparse:
        lower *= rng.random((n, n)) < 0.1
    a = lower + lower.T + np.diag(np.arange(841.0, 1001.0))
    ref, basis = la.eigh(a)
    selected = np.arange(n - k, n) if which == "LA" else np.arange(k)
    solver = KrylovSchur(SparseOperator(matrix(a)), k, ncv=40, which=which, tol=1e-12)
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        residual = a @ v - v * w
        assert np.max(abs(residual)) < 1e-10
        np.testing.assert_allclose(np.sort(w), ref[selected], atol=1e-10, rtol=0)
        np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-11, rtol=0)
        assert np.linalg.norm(v - basis[:, selected] @ (basis[:, selected].T @ v), 2) < 1e-10
        relative = np.linalg.norm(residual, axis=0) / np.maximum(1.0, abs(w))
        np.testing.assert_allclose(solver.residuals.numpy(), relative, atol=1e-14, rtol=0)
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
    dot = tmp_path / "symmetric.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


def test_normalized_convergence_does_not_imply_absolute_accuracy():
    # Deliberately inexact unit vectors demonstrate the contract at two scales.
    angle = 1e-9
    vectors = wp.array([[np.cos(angle), np.sin(angle)]], dtype=wp.float64)
    for scale in (1.0, 1000.0):
        a = scale * np.diag([1.0, 2.0])
        evaluation = EigenpairEvaluation(SparseOperator(matrix(a)), vectors, tol=2e-9)
        evaluation.run()
        v, w = vectors.numpy().T, evaluation.values.numpy()
        residual = a @ v - v * w
        np.testing.assert_allclose(evaluation.residuals.numpy(), [1e-9], rtol=1e-6)
        assert evaluation.converged.numpy()[0] == 1
        np.testing.assert_allclose(np.max(abs(residual)), scale * 1e-9, rtol=1e-6)
        if scale == 1000.0:
            assert np.max(abs(residual)) > 1e-7
