"""Explicit sparse indefinite factors must resolve the original requested modes."""

import re
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


@pytest.mark.parametrize(
    "options",
    [dict(reordering="colamd", refinement_steps=0), dict(matching=True, refinement_steps=1)],
)
@pytest.mark.parametrize(
    "key,k,ncv", [("SymEigs_17", 3, 6), ("SymEigs_22", 10, 20), ("SymEigs_27", 20, 50)]
)
def test_sparse_indefinite_original_modes(key, k, ncv, options, tmp_path):
    data = np.load(Path(__file__).parent / "fixtures/spectra_sparse_sm.npz")
    a = data[key]
    values, basis = np.linalg.eigh(a)
    selected = np.argsort(abs(values))[:k]
    inv = CuDSSInverse(matrix(a), 1, mtype="general", **options)
    solver = KrylovSchur(inv, k, ncv=ncv, tol=1e-12)
    evaluation = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-12)
    graph = solver.capture(100, finalizer=evaluation.run)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = evaluation.values.numpy(), solver.eigenvectors.numpy().T
            np.testing.assert_allclose(np.sort(w), np.sort(values[selected]), atol=1e-9, rtol=0)
            assert abs(a @ v - v * w).max() < 1e-9
            np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                np.linalg.svd(basis[:, selected].T @ v, compute_uv=False),
                np.ones(k),
                atol=1e-9,
                rtol=0,
            )
            assert solver.converged.numpy()[0] == evaluation.converged.numpy()[0] == k
        dot = tmp_path / "pivoted.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        text = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', text) and "DtoH" not in text
    finally:
        solver.graph = graph = None
        inv.close()


@pytest.mark.parametrize(
    "options", [dict(reordering="colamd", refinement_steps=0), dict(matching=True)]
)
def test_pivoted_inverse_in_place_changed_rhs(options):
    a = np.load(Path(__file__).parent / "fixtures/cupy_sm.npz")["a"]
    shift = -np.sqrt(np.finfo(float).eps) * abs(a).max()
    a = a - shift * np.eye(len(a))
    inv = CuDSSInverse(matrix(a), 3, mtype="general", **options)
    rhs = wp.ones((3, len(a)), dtype=wp.float64)
    source = wp.empty_like(rhs)
    inv.apply(rhs, rhs)
    with wp.ScopedCapture() as cap:
        wp.copy(rhs, source)
        inv.apply(rhs, rhs)
    graph = cap.graph
    try:
        for seed in (1, 2):
            b = np.random.default_rng(seed).normal(size=rhs.shape)
            wp.copy(source, wp.array(b, dtype=wp.float64))
            wp.capture_launch(graph)
            assert np.linalg.norm(rhs.numpy() @ a - b) / np.linalg.norm(b) < 1e-13
    finally:
        cap.graph = graph = None
        inv.close()


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_matching_nonsymmetric_scaling_and_changed_rhs(scale):
    # A column permutation forces nontrivial matching; scaling must be undone
    # in the returned solution, including captured in-place multiple RHS solves.
    n = 24
    a = np.diag(np.linspace(2, 3, n)) + np.diag(np.full(n - 1, 0.3), 1)
    a = scale * a[:, np.roll(np.arange(n), 7)]
    inv = CuDSSInverse(matrix(a), 3, mtype="general", matching=True)
    rhs = wp.ones((3, n), dtype=wp.float64)
    source = wp.empty_like(rhs)
    inv.apply(rhs, rhs)
    with wp.ScopedCapture() as cap:
        wp.copy(rhs, source)
        inv.apply(rhs, rhs)
    graph = cap.graph
    try:
        for seed in (3, 4):
            b = np.random.default_rng(seed).normal(size=(3, n))
            wp.copy(source, wp.array(b, dtype=wp.float64))
            wp.capture_launch(graph)
            x = rhs.numpy()
            assert np.linalg.norm(x @ a.T - b) / np.linalg.norm(b) < 1e-13
            np.testing.assert_allclose(x, np.linalg.solve(a, b.T).T, rtol=1e-12, atol=1e-15)
    finally:
        cap.graph = graph = None
        inv.close()
