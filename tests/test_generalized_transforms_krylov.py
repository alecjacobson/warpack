"""Original generalized fixtures: physical accuracy and requested invariant subspace."""

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator


@pytest.mark.parametrize(
    "key,k,ncv,sigma,mode",
    [
        ("SymGEigsCholesky_17", 3, 6, 0.0, "shift-invert"),
        ("SymGEigsCholesky_22", 10, 20, 0.0, "shift-invert"),
        ("SymGEigsCholesky_27", 20, 50, 0.0, "shift-invert"),
        ("SymGEigsShift_5", 10, 20, 1.2345, "shift-invert"),
        ("SymGEigsShift_45", 10, 20, 1.2345, "cayley"),
    ],
)
def test_original_generalized_requested_modes(key, k, ncv, sigma, mode):
    folder = Path(__file__).parent / "fixtures"
    if key == "SymGEigsShift_5":
        data = np.load(folder / "spectra_generalized_shift_5.npz")
        a, b = data["a"], data["b"]
    else:
        data = np.load(folder / "spectra_generalized_matched.npz")
        a, b = data[key + "_a"], data[key + "_b"]
    values, basis = la.eigh(a, b)
    transformed = (values + sigma) / (values - sigma) if mode == "cayley" else 1 / (values - sigma)
    selected = np.argsort(-abs(transformed))[:k]
    inverse = CuDSSInverse(
        matrix(a - sigma * b), 1, mtype="general", matching=True, refinement_steps=1
    )
    solver = GeneralizedKrylovSchur(
        SparseOperator(matrix(a)),
        SparseOperator(matrix(b)),
        inverse,
        k,
        ncv=ncv,
        target=sigma,
        mode=mode,
        tol=1e-12,
    )
    graph = solver.capture(20)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            np.testing.assert_allclose(np.sort(w), np.sort(values[selected]), atol=1e-9, rtol=0)
            assert abs(a @ v - (b @ v) * w).max() < 1e-9
            np.testing.assert_allclose(v.T @ b @ v, np.eye(k), atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                la.svdvals(basis[:, selected].T @ b @ v), np.ones(k), atol=1e-9, rtol=0
            )
            assert solver.converged.numpy()[0] == k
    finally:
        solver.graph = graph = None
        inverse.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_cayley_selection_differs_from_nearest_physical_modes(sign):
    n = 30
    rng = np.random.default_rng(37)
    q = la.qr(rng.normal(size=(n, n)))[0]
    t = np.diag(np.linspace(0.5, 2.0, n)) + np.diag(np.full(n - 1, 0.1), 1)
    spectrum = sign * np.r_[-0.2, 0.5, 4.0, np.arange(8.0, 35.0)]
    b = t.T @ t
    a = t.T @ (q * spectrum) @ q.T @ t
    sigma = sign
    # Nearest-shift would include -0.2; Cayley must include 4 instead.
    expected = sign * np.array([0.5, 4.0])
    inverse = CuDSSInverse(matrix(a - sigma * b), 1, mtype="general", matching=True)
    solver = GeneralizedKrylovSchur(
        SparseOperator(matrix(a)),
        SparseOperator(matrix(b)),
        inverse,
        2,
        ncv=12,
        target=sigma,
        mode="cayley",
        tol=1e-11,
    )
    graph = solver.capture(20)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            np.testing.assert_allclose(np.sort(w), np.sort(expected), atol=1e-10, rtol=0)
            assert abs(a @ v - (b @ v) * w).max() < 1e-10
            np.testing.assert_allclose(v.T @ b @ v, np.eye(2), atol=1e-11, rtol=0)
            assert solver.converged.numpy()[0] == 2
    finally:
        solver.graph = graph = None
        inverse.close()
