"""Requested-spectrum regressions: a small residual alone is insufficient."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import KrylovSchur, SparseOperator


@pytest.mark.parametrize("zero_operator", [False, True])
def test_spectra_explicit_nullspace_start(zero_operator):
    from pathlib import Path

    data = np.load(Path(__file__).parent / "fixtures/spectra_null_start.npz")
    a = np.zeros_like(data["a"]) if zero_operator else data["a"]
    start = data["v0"].ravel()
    assert np.linalg.norm(a @ start) < 1e-12
    solver = KrylovSchur(
        SparseOperator(matrix(a)),
        3,
        ncv=6,
        which="LA",
        tol=1e-12,
        initial_vector=wp.array(start, dtype=wp.float64),
    )
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        assert_pairs(a, solver, np.linalg.eigvalsh(a)[-3:])


def cycle(n=20):
    eye = np.eye(n)
    return eye - 0.5 * np.roll(eye, 1, axis=0) - 0.5 * np.roll(eye, -1, axis=0)


def assert_pairs(a, solver, expected):
    values = solver.eigenvalues.numpy()
    vectors = solver.eigenvectors.numpy().T
    np.testing.assert_allclose(np.sort(values), expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(solver.k), atol=1e-10, rtol=0)
    assert np.max(abs(a @ vectors - vectors * values)) < 1e-9
    assert solver.converged.numpy()[0] == solver.k
    # If k cuts through a repeated eigenvalue, any subspace inside that entire
    # boundary eigenspace is valid; signs and individual reference vectors are not.
    ref, basis = np.linalg.eigh(a)
    allowed = basis[:, ref >= expected[0] - 1e-9]
    np.testing.assert_allclose(
        np.linalg.svd(allowed.T @ vectors, compute_uv=False), np.ones(solver.k), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("k,ncv", [(3, 6), (5, 12), (6, 12)])
@pytest.mark.parametrize("execution", ["eager", "adaptive", "fixed"])
def test_spectra_cycle_requested_multiplicities(k, ncv, execution):
    a = cycle()
    expected = np.sort(2 * np.sin(np.pi * np.arange(20) / 20) ** 2)[-k:]
    solver = KrylovSchur(SparseOperator(matrix(a)), k, ncv=ncv, tol=1e-12)
    if execution == "eager":
        solver.solve(100)
        assert_pairs(a, solver, expected)
    else:
        graph = solver.capture(100, adaptive=execution == "adaptive")
        for _ in range(2):
            wp.capture_launch(graph)
            assert_pairs(a, solver, expected)


@pytest.mark.parametrize("multiplicity", [2, 4, 8])
@pytest.mark.parametrize("compact", [False, True])
def test_rotated_repeated_clusters(multiplicity, compact):
    n, k = 80, multiplicity + 1
    spectrum = np.r_[
        np.linspace(-1, 6, n - 2 * multiplicity),
        np.full(multiplicity, 8.0),
        np.full(multiplicity, 10.0),
    ]
    rng = np.random.default_rng(123 + multiplicity)
    rotation = np.linalg.qr(rng.normal(size=(n, n)))[0]
    a = (rotation * spectrum) @ rotation.T
    start = wp.array(rng.normal(size=n), dtype=wp.float64)
    solver = KrylovSchur(
        SparseOperator(matrix(a)), k, ncv=2 * k if compact else 32, tol=1e-11, initial_vector=start
    )
    graph = solver.capture(100)
    for seed in [0, 1]:
        wp.copy(start, wp.array(np.random.default_rng(seed).normal(size=n), dtype=wp.float64))
        wp.capture_launch(graph)
        assert_pairs(a, solver, spectrum[-k:])


def test_full_space_does_not_require_redundant_exploration():
    a = np.diag(np.arange(1.0, 13.0))
    solver = KrylovSchur(SparseOperator(matrix(a)), 4, ncv=12, tol=1e-11)
    wp.capture_launch(solver.capture(20))
    assert_pairs(a, solver, np.arange(9.0, 13.0))
    assert solver.iterations.numpy()[0] == 1


def test_transformed_sm_retains_physical_accuracy():
    """A count-only upstream failure must not become a physical residual failure."""
    from pathlib import Path

    from warpack import CuDSSInverse, EigenpairEvaluation

    a = np.load(Path(__file__).parent / "fixtures/spectra_transformed_sm.npz")["a"]
    inverse = CuDSSInverse(matrix(a - np.eye(len(a))), 1, mtype="symmetric")
    solver = KrylovSchur(inverse, 3, ncv=6, which="SM", tol=1e-12)
    evaluation = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-9)
    graph = solver.capture(100, finalizer=evaluation.run)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            values, vectors = evaluation.values.numpy(), solver.eigenvectors.numpy().T
            expected = np.linalg.eigvalsh(a)
            expected = expected[np.argsort(-abs(expected - 1))[:3]]
            np.testing.assert_allclose(np.sort(values), np.sort(expected), atol=1e-9, rtol=0)
            assert np.max(abs(a @ vectors - vectors * values)) < 1e-9
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-10, rtol=0)
            assert evaluation.converged.numpy()[0] == 3
    finally:
        solver.graph = graph = None
        inverse.close()
