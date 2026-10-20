"""Installed fast-block composition, replay, ownership and metadata contracts."""

import weakref

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG


@pytest.mark.parametrize("bucket", [0, 2])
def test_installed_block_replays_current_pencil_and_drops_owner(bucket, tmp_path):
    if not hasattr(linear.LinearOperator, "batch_offsets"):
        pytest.skip("upstream batched CG required")
    n, k, width = 64, 4, 8
    rng = np.random.default_rng(183)
    q = la.qr(rng.normal(size=(n, n)))[0]
    lower = np.eye(n) + np.diag(np.full(n - 1, 0.1), -1)
    mass = lower @ lower.T
    v = la.solve_triangular(lower.T, q, lower=False)
    expected = np.r_[1, 2, 3, 3, np.arange(4, n)]
    host = mass @ (v * expected) @ v.T @ mass
    a, b = matrix(host), matrix(mass)
    cg = BatchedCGPreconditioner(
        a, width, tol=0.01, maxiter=25, working_dtype=wp.float32, matrix_layout="columns"
    )
    start = wp.array(rng.normal(size=(width, n)), dtype=wp.float64)
    solver = BlockLOBPCG(
        BatchedSparseOperator(a),
        k,
        width,
        mass=BatchedSparseOperator(b),
        preconditioner=cg,
        initial_vectors=start,
        tol=1e-10,
        reuse_operator_images=True,
        search_bucket_size=2,
        keep_history=True,
        preconditioner_bucket_size=bucket,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
    )
    graph = solver.capture(150)
    source = start.numpy().copy()
    for scale in [1.0, 1.5, 1.0]:
        a.values.assign((host * scale).ravel())
        wp.capture_launch(graph)
        w, x = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        np.testing.assert_allclose(w, scale * expected[:k], atol=1e-9)
        np.testing.assert_allclose(x.T @ mass @ x, np.eye(k), atol=1e-10)
        assert la.norm(x - v[:, :k] @ (v[:, :k].T @ mass @ x)) < 1e-8
        assert la.norm(scale * host @ x - (mass @ x) * w, axis=0).max() < 1e-8
        np.testing.assert_array_equal(start.numpy(), source)
    dot = tmp_path / "installed-block.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
    owner = weakref.ref(solver)
    graph = None
    del solver
    assert owner() is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"width": True},
        {"width": 2.5},
        {"maxiter": 0},
        {"maxiter": True},
        {"maxiter": 2.5},
        {"tol": -1},
        {"tol": np.nan},
        {"atol": np.inf},
        {"atol": -1},
    ],
)
def test_batched_invalid_metadata(kwargs):
    if not hasattr(linear.LinearOperator, "batch_offsets"):
        pytest.skip("upstream batched CG required")
    args = dict(width=4)
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        BatchedCGPreconditioner(matrix(np.eye(32)), **args)


def test_stock_runtime_rejects_missing_batched_cg():
    if hasattr(linear.LinearOperator, "batch_offsets"):
        pytest.skip("stock-runtime negative control")
    with pytest.raises(RuntimeError, match="batched"):
        BatchedCGPreconditioner(matrix(np.eye(32)), 4)


def test_block_eager_iteration_counter():
    solver = BlockLOBPCG(BatchedSparseOperator(matrix(np.diag(np.arange(1.0, 33)))), 2, 4)
    solver.solve(2)
    assert solver.iterations.numpy()[0] == 2
    graph = solver.capture(3, adaptive=False)
    wp.capture_launch(graph)
    assert solver.iterations.numpy()[0] == 3
    solver.solve(1)
    assert solver.iterations.numpy()[0] == 1


@pytest.mark.parametrize("bucket", [0, 2])
def test_batched_width_mismatch_is_rejected_at_setup(bucket):
    if not hasattr(linear.LinearOperator, "batch_offsets"):
        pytest.skip("upstream batched CG required")
    a = matrix(np.eye(32))
    preconditioner = BatchedCGPreconditioner(a, 3)
    with pytest.raises(ValueError, match="width must match"):
        BlockLOBPCG(
            BatchedSparseOperator(a),
            2,
            4,
            preconditioner=preconditioner,
            preconditioner_bucket_size=bucket,
        )
