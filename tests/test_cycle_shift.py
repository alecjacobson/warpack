"""Cycle low modes: original-equation accuracy differs from inverse-space counts."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


@pytest.mark.parametrize("k,ncv", [(3, 6), (5, 12), (6, 12)])
def test_captured_cycle_shift_original_modes(k, ncv):
    n, sigma = 20, -1e-6
    eye = np.eye(n)
    a = eye - 0.5 * np.roll(eye, 1, axis=0) - 0.5 * np.roll(eye, -1, axis=0)
    expected = np.sort(2 * np.sin(np.pi * np.arange(n) / n) ** 2)[:k]
    reference, basis = np.linalg.eigh(a)
    allowed = basis[:, reference <= expected[-1] + 1e-9]
    inverse = CuDSSInverse(matrix(a - sigma * eye), 1, mtype="symmetric")
    solver = KrylovSchur(inverse, k, ncv=ncv, tol=1e-12)
    evaluation = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-9)
    graph = solver.capture(100, finalizer=evaluation.run)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            values, vectors = evaluation.values.numpy(), solver.eigenvectors.numpy().T
            np.testing.assert_allclose(np.sort(values), expected, atol=1e-9, rtol=0)
            assert np.max(abs(a @ vectors - vectors * values)) < 1e-9
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(k), atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                np.linalg.svd(allowed.T @ vectors, compute_uv=False),
                np.ones(k),
                atol=1e-10,
                rtol=0,
            )
            assert evaluation.converged.numpy()[0] == k
            # Do not substitute the inverse-space counter for physical accuracy.
            assert solver.converged.numpy()[0] == np.count_nonzero(
                solver.residuals.numpy() <= np.float32(solver.tol)
            )
    finally:
        solver.graph = graph = None
        inverse.close()
