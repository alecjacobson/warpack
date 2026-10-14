"""Indefinite inverse accuracy is checked before and inside the eigensolver."""

import re
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


def fixture():
    data = np.load(Path(__file__).parent / "fixtures/cupy_sm.npz")
    a, v0 = data["a"], data["v0"]
    sigma = -np.sqrt(np.finfo(float).eps) * abs(a).max()
    return a, v0, sigma


@pytest.mark.parametrize("mtype", ["symmetric", "general"])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("in_place", [False, True])
def test_refined_inverse_replay_and_changed_rhs(mtype, width, in_place, tmp_path):
    a, _, sigma = fixture()
    shifted = a - sigma * np.eye(len(a))
    inv = CuDSSInverse(
        matrix(shifted), width, mtype=mtype, reordering="default", refinement_steps=1
    )
    rng = np.random.default_rng(42)
    rhs = wp.array(rng.normal(size=(width, len(a))), dtype=wp.float64)
    source = wp.empty_like(rhs)
    out = rhs if in_place else wp.empty_like(rhs)
    inv.apply(rhs, out)
    with wp.ScopedCapture() as cap:
        wp.copy(rhs, source)
        inv.apply(rhs, out)
    graph = cap.graph
    try:
        for _ in range(2):
            b = rng.normal(size=(width, len(a)))
            wp.copy(source, wp.array(b, dtype=wp.float64))
            wp.capture_launch(graph)
            x = out.numpy()
            assert np.linalg.norm(x @ shifted.T - b) / np.linalg.norm(b) < 1e-13
            np.testing.assert_allclose(x, np.linalg.solve(shifted, b.T).T, atol=1e-13, rtol=1e-13)
        dot = tmp_path / "refined.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        text = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', text) and "DtoH" not in text
    finally:
        cap.graph = graph = None
        inv.close()


def test_refinement_accepts_block_csr():
    import warp.sparse as ws

    a, _, sigma = fixture()
    a = a - sigma * np.eye(len(a))
    n = len(a) // 2
    blocks = a.reshape(n, 2, n, 2).transpose(0, 2, 1, 3).reshape(n * n, 2, 2).copy()
    block_matrix = ws.bsr_from_triplets(
        n,
        n,
        wp.array(np.repeat(np.arange(n), n), dtype=wp.int32),
        wp.array(np.tile(np.arange(n), n), dtype=wp.int32),
        wp.array(blocks, dtype=wp.mat22d),
    )
    inv = CuDSSInverse(block_matrix, 2, mtype="symmetric", refinement_steps=2)
    b = np.random.default_rng(43).normal(size=(2, len(a)))
    rhs = wp.array(b, dtype=wp.float64)
    out = wp.empty_like(rhs)
    try:
        inv.apply(rhs, out)
        assert np.linalg.norm(out.numpy() @ a - b) / np.linalg.norm(b) < 1e-13
    finally:
        inv.close()


@pytest.mark.parametrize("k", [3, 6, 12])
@pytest.mark.parametrize("options", [dict(mtype="symmetric"), dict(mtype="general", matching=True)])
def test_cupy_sm_original_residual_and_spectrum(k, options):
    a, v0, sigma = fixture()
    inv = CuDSSInverse(matrix(a - sigma * np.eye(len(a))), 1, **options)
    solver = KrylovSchur(
        inv, k, ncv=len(a), tol=1e-14, initial_vector=wp.array(v0, dtype=wp.float64)
    )
    evaluation = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-12)
    graph = solver.capture(1, finalizer=evaluation.run)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = evaluation.values.numpy(), solver.eigenvectors.numpy().T
            assert np.linalg.norm(a @ v - v * w) / np.linalg.norm(a) < 1e-12
            expected, basis = np.linalg.eigh(a)
            selected = np.argsort(abs(expected))[:k]
            np.testing.assert_allclose(np.sort(w), np.sort(expected[selected]), atol=1e-12, rtol=0)
            np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                np.linalg.svd(basis[:, selected].T @ v, compute_uv=False), np.ones(k), atol=1e-12
            )
            assert evaluation.converged.numpy()[0] == k
    finally:
        solver.graph = graph = None
        inv.close()


@pytest.mark.parametrize("shift", [1e-4, 1e-6])
def test_refined_near_null_generalized_modes(shift):
    from warpack import GeneralizedKrylovSchur

    n, k = 48, 4
    rng = np.random.default_rng(246)
    rotation = np.linalg.qr(rng.normal(size=(n, n)))[0]
    spectrum = np.r_[0.0, 0.0, 1e-8, 1e-5, np.linspace(0.1, 4.0, n - k)]
    transform = np.diag(np.linspace(0.5, 2.0, n)) + np.diag(np.full(n - 1, 0.1), 1)
    mass = transform.T @ transform
    stiffness = transform.T @ (rotation * spectrum) @ rotation.T @ transform
    reference = np.linalg.solve(transform, rotation[:, :k])
    inv = CuDSSInverse(matrix(stiffness + shift * mass), 1, refinement_steps=1)
    solver = GeneralizedKrylovSchur(
        SparseOperator(matrix(stiffness)),
        SparseOperator(matrix(mass)),
        inv,
        k,
        ncv=24,
        target=-shift,
        tol=1e-10,
    )
    graph = solver.capture(20)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            np.testing.assert_allclose(np.sort(w), spectrum[:k], atol=1e-10, rtol=0)
            assert np.max(abs(stiffness @ v - (mass @ v) * w)) < 1e-9
            np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                np.linalg.svd(reference.T @ mass @ v, compute_uv=False),
                np.ones(k),
                atol=1e-9,
                rtol=0,
            )
            assert solver.converged.numpy()[0] == k
    finally:
        solver.graph = graph = None
        inv.close()
