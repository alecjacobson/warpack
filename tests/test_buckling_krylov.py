"""Buckling selection is transformed magnitude, not nearest physical lambda."""

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import BucklingEigensolver, CuDSSInverse, SparseOperator


def check_problem(a, b, sigma, k, ncv, cycles):
    mu, basis = la.eigh(b, a)
    spectrum = 1 / mu
    selected = np.argsort(-abs(spectrum / (spectrum - sigma)))[:k]
    inv = CuDSSInverse(matrix(a - sigma * b), 1, mtype="general", matching=True, refinement_steps=1)
    solver = BucklingEigensolver(
        SparseOperator(matrix(a)),
        SparseOperator(matrix(b)),
        inv,
        k,
        sigma,
        ncv=ncv,
        tol=1e-12,
        method="krylov",
    )
    graph = solver.capture(cycles)
    try:
        for _ in range(3):
            wp.capture_launch(graph)
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            np.testing.assert_allclose(np.sort(w), np.sort(spectrum[selected]), atol=1e-9, rtol=0)
            assert abs(a @ v - (b @ v) * w).max() < 1e-9
            np.testing.assert_allclose(v.T @ a @ v, np.eye(k), atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                la.svdvals(basis[:, selected].T @ a @ v), np.ones(k), atol=1e-9, rtol=0
            )
            assert solver.converged.numpy()[0] == k
    finally:
        solver.core.graph = solver.graph = graph = None
        inv.close()


@pytest.mark.parametrize("sigma", [-2.3, 2.3])
def test_buckling_krylov_signed_shifts(sigma):
    a = np.diag(np.arange(1.0, 31.0))
    b = np.diag(np.where(np.arange(30) % 2, -1.0, 1.0))
    check_problem(a, b, sigma, 4, 20, 20)


def test_original_spectra_buckling_30():
    data = np.load(Path(__file__).parent / "fixtures/spectra_buckling_30.npz")
    check_problem(data["a"], data["b"], 1.2345, 10, 20, 20)
