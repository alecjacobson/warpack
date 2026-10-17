"""General SM: inverse accuracy, budget and requested modes are distinct gates."""

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.upstream.adapters import complex_matrix, output
from warpack import ShiftInvertEigensolver


@pytest.mark.parametrize(
    "key",
    ["GenEigs_3", "GenEigs_21", "GenEigs_27", "ComplexEigs_3", "ComplexEigs_21", "ComplexEigs_27"],
)
def test_spectra_general_sm_validated_budget(key):
    a = np.load(Path(__file__).parent / "fixtures/spectra_general_sm.npz")[key]
    n = len(a)
    k, ncv, budget = (3, 6, 200) if n == 10 else (10, 30, 30)
    solver = ShiftInvertEigensolver(complex_matrix(a), k, ncv=ncv, tol=1e-12)
    graph = None
    try:
        real = np.block([[a.real, -a.imag], [a.imag, a.real]])
        rhs_host = np.random.default_rng(91).normal(size=(1, 2 * n))
        rhs = wp.array(rhs_host, dtype=wp.float64)
        solution = wp.empty_like(rhs)
        solver.inverse.apply(rhs, solution)
        assert (
            np.linalg.norm(solution.numpy() @ real.T - rhs_host) / np.linalg.norm(rhs_host) < 1e-12
        )

        ref, basis = la.eig(a)
        selected = np.flatnonzero(abs(ref) <= np.sort(abs(ref))[k - 1] + 1e-9)
        allowed = la.orth(basis[:, selected])
        graph = solver.capture(budget)
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
            assert np.max(abs(a @ v - v * w)) < 1e-9
            costs = abs(w[:, None] - ref[selected])
            rows, cols = linear_sum_assignment(costs)
            assert len(rows) == k and np.max(costs[rows, cols]) < 1e-9
            assert np.linalg.norm(v - allowed @ (allowed.conj().T @ v), 2) < 1e-9
            np.testing.assert_allclose(np.linalg.norm(v, axis=0), np.ones(k), atol=1e-12)
            # Nonsymmetric right eigenvectors need not be mutually orthogonal.
            assert solver.status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == k
    finally:
        solver.graph = solver.core.graph = graph = None
        solver.close()
