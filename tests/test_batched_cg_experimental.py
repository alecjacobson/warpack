"""Upstream batched CG retains independent RHS convergence and captured inputs."""

import re

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.experimental.batched_cg import BatchedCGInverse, RepeatedPreconditioner
from benchmarks.upstream.adapters import matrix

pytestmark = pytest.mark.skipif(
    not hasattr(linear.LinearOperator, "batch_offsets"), reason="upstream batched CG required"
)


def fixture(block):
    n = 45
    a = np.diag(np.linspace(2, 5, n)) + np.diag(np.full(n - 1, -0.4), 1)
    a += np.triu(a, 1).T
    if block == 1:
        return a, matrix(a)
    b = sp.csr_matrix(a).tobsr(blocksize=(3, 3))
    wa = ws.bsr_from_triplets(
        n // 3,
        n // 3,
        wp.array(np.repeat(np.arange(n // 3), np.diff(b.indptr)), dtype=wp.int32),
        wp.array(b.indices, dtype=wp.int32),
        wp.array(b.data, dtype=wp.mat33d),
    )
    return a, wa


@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("preconditioner", ["none", "fsai", "bj"])
@pytest.mark.parametrize("in_place", [False, True])
def test_batched_cg_scaled_and_zero_rhs_replay(block, preconditioner, in_place, tmp_path):
    a, wa = fixture(block)
    if preconditioner == "fsai":
        if not hasattr(linear, "FSAI"):
            pytest.skip("pinned upstream FSAI required")
        pre = linear.FSAI(wa, max_row_size=8, factor_dtype=wp.float32)
    elif preconditioner == "bj":
        pre = linear.preconditioner(wa, ptype="block_jacobi_sequential")
    else:
        pre = None
    width = 7
    solver = BatchedCGInverse(wa, width, preconditioner=pre, tol=1e-12, maxiter=100)
    source = wp.zeros((width, len(a)), dtype=wp.float64)
    out = source if in_place else wp.empty_like(source)
    solver.apply(source, out)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.reset_stats()
        solver.apply(source, out)
    rng = np.random.default_rng(199)
    for _ in range(2):
        b = rng.normal(size=source.shape) * np.array([1, 1e-8, 1e8, 0, 2, 0.01, 0.001])[:, None]
        source.assign(b)
        wp.capture_launch(capture.graph)
        x = out.numpy()
        norms = np.linalg.norm(b, axis=1)
        residual = np.linalg.norm(x @ a.T - b, axis=1)
        assert np.max(residual[norms > 0] / norms[norms > 0]) < 2e-12
        np.testing.assert_allclose(x[3], 0, atol=0)
        expected = np.linalg.solve(a, b.T).T
        assert np.max(np.linalg.norm(x - expected, axis=1)[norms > 0] / norms[norms > 0]) < 2e-12
        if not in_place:
            np.testing.assert_array_equal(source.numpy(), b)
        assert solver.stats.numpy()[2:].tolist() == [0, width]
    dot = tmp_path / "batch-cg.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_batched_fsai_application_matches_upstream(dtype):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    a, wa = fixture(3)
    pre = linear.FSAI(wa, max_row_size=8, factor_dtype=dtype)
    offsets = wp.array([j * len(a) for j in range(8)], dtype=wp.int32)
    batched = RepeatedPreconditioner(pre, 7, len(a), offsets, wa.device)
    rng = np.random.default_rng(51)
    x, y = [wp.array(rng.normal(size=7 * len(a)), dtype=wp.float64) for _ in range(2)]
    out, expected = wp.empty_like(x), wp.empty_like(x)
    batched.matvec(x, y, out, 0.75, -0.25)
    for j in range(7):
        sl = slice(j * len(a), (j + 1) * len(a))
        pre.matvec(x[sl], y[sl], expected[sl], 0.75, -0.25)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("in_place", [False, True])
def test_fp32_inner_solve_keeps_scaled_rhs_independent(block, in_place):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    a, wa = fixture(block)
    pre = linear.FSAI(wa, max_row_size=8, factor_dtype=wp.float32)
    solver = BatchedCGInverse(
        wa, 7, preconditioner=pre, tol=1e-6, maxiter=100, working_dtype=wp.float32
    )
    source = wp.zeros((7, len(a)), dtype=wp.float64)
    out = source if in_place else wp.empty_like(source)
    solver.apply(source, out)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.reset_stats()
        solver.apply(source, out)
    for seed in (2, 3):
        b = np.random.default_rng(seed).normal(size=source.shape)
        b *= np.array([1, 1e-8, 1e8, 0, 2, 0.01, 0.001])[:, None]
        source.assign(b)
        wp.capture_launch(capture.graph)
        norms = np.linalg.norm(b, axis=1)
        residual = np.linalg.norm(out.numpy() @ a.T - b, axis=1)
        assert np.max(residual[norms > 0] / norms[norms > 0]) < 2e-6
        np.testing.assert_array_equal(out.numpy()[3], 0)
        assert solver.stats.numpy()[2:].tolist() == [0, 7]


@pytest.mark.parametrize("coupled_mass", [False, True])
def test_fp32_preconditioning_preserves_fp64_eigen_accuracy(coupled_mass):
    import scipy.linalg as la

    from benchmarks.experimental.block_lobpcg import BlockLOBPCG
    from warpack import SparseOperator

    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    n, k, width = 80, 4, 8
    mass = np.diag(np.linspace(1, 2, n))
    if coupled_mass:
        mass += np.diag(np.full(n - 1, 0.1), 1) + np.diag(np.full(n - 1, 0.1), -1)
    a = np.diag(np.linspace(1, 80, n))
    a += np.diag(np.full(n - 1, -0.1), 1) + np.diag(np.full(n - 1, -0.1), -1)
    shifted = matrix(a + mass)
    pre = linear.FSAI(shifted, max_row_size=8, factor_dtype=wp.float32)
    inverse = BatchedCGInverse(
        shifted, width, preconditioner=pre, tol=0.01, maxiter=3, working_dtype=wp.float32
    )
    solver = BlockLOBPCG(
        SparseOperator(matrix(a)),
        k,
        width,
        mass=SparseOperator(matrix(mass)),
        preconditioner=inverse,
        tol=1e-10,
    )
    values, basis = la.eigh(a, mass)
    graph = solver.capture(60)
    for _ in range(2):
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        np.testing.assert_allclose(w, values[:k], atol=1e-9, rtol=0)
        assert np.max(abs(a @ v - (mass @ v) * w)) < 1e-8
        np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), atol=1e-10)
        assert la.norm(v - basis[:, :k] @ (basis[:, :k].T @ mass @ v)) < 1e-8
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_batched_cg_requires_compact_sparse_rows(dtype):
    _, wa = fixture(3)
    # Model padded row storage: the adapter must reject it before reading slots
    # beyond row_counts, in either inner precision.
    wa.row_counts = wp.zeros(wa.nrow, dtype=wp.int32, device=wa.device)
    with pytest.raises(ValueError, match="bsr_compress"):
        BatchedCGInverse(wa, 7, working_dtype=dtype)
