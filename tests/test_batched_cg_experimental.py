"""Upstream batched CG retains independent RHS convergence and captured inputs."""

import re

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.upstream.adapters import matrix
from warpack.batched_cg import BatchedCGPreconditioner as BatchedCGInverse
from warpack.batched_cg import RepeatedPreconditioner

pytestmark = pytest.mark.skipif(
    not all(hasattr(linear.LinearOperator, attr) for attr in ("batch_offsets", "max_batch_length")),
    reason="upstream batched CG required",
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
@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_batched_fsai_application_matches_upstream(layout, dtype):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    a, wa = fixture(3)
    pre = linear.FSAI(wa, max_row_size=8, factor_dtype=dtype)
    offsets = wp.array([j * len(a) for j in range(8)], dtype=wp.int32)
    batched = RepeatedPreconditioner(pre, 7, len(a), offsets, wa.device, layout=layout)
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
@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_fp32_inner_solve_keeps_scaled_rhs_independent(layout, block, in_place):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    a, wa = fixture(block)
    pre = linear.FSAI(wa, max_row_size=8, factor_dtype=wp.float32)
    solver = BatchedCGInverse(
        wa,
        7,
        preconditioner=pre,
        tol=1e-6,
        maxiter=100,
        working_dtype=wp.float32,
        fsai_layout=layout,
        matrix_layout=layout,
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
@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_fp32_preconditioning_preserves_fp64_eigen_accuracy(layout, coupled_mass):
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
        shifted,
        width,
        preconditioner=pre,
        tol=0.01,
        maxiter=3,
        working_dtype=wp.float32,
        fsai_layout=layout,
        matrix_layout=layout,
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


@pytest.mark.parametrize("case", ["torus", "disconnected", "weak_positive"])
def test_batched_cg_adaptive_hodge_nullspace_and_positive_guards(case, tmp_path):
    from benchmarks.applications.hodge import make_hodge_case
    from benchmarks.applications.runner import accuracy
    from benchmarks.experimental.block_kernels import BatchedSparseOperator
    from benchmarks.experimental.block_lobpcg import BlockLOBPCG

    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = make_hodge_case(case, side=4).problem
    shifted = matrix(p.a - p.target * p.b)
    fsai = linear.FSAI(shifted, max_row_size=16, factor_dtype=wp.float32)
    inverse = BatchedCGInverse(
        shifted, 8, preconditioner=fsai, tol=0.01, maxiter=10, working_dtype=wp.float32
    )
    if inverse.requires_static_outer_capture:
        pytest.skip("upstream CG must preallocate conditional-loop control")
    starts = wp.zeros((8, p.n), dtype=wp.float64)
    rng = np.random.default_rng(681)
    starts.assign(rng.normal(size=starts.shape))
    solver = BlockLOBPCG(
        BatchedSparseOperator(matrix(p.a)),
        p.k,
        8,
        mass=BatchedSparseOperator(matrix(p.b)),
        preconditioner=inverse,
        initial_vectors=starts,
        tol=1e-12,
    )
    graph = solver.capture(200)
    for _ in range(2):
        starts.assign(rng.normal(size=starts.shape))
        wp.capture_launch(graph)
        check = accuracy(p, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert check["accepted"], check
        assert check["recovered_nullity"] == p.nullity
        assert check["positive_guard"] > 0
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == p.k
        assert 0 < solver.iterations.numpy()[0] < 200
    dot = tmp_path / "hodge-batched-adaptive.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("dtype,tolerance", [(wp.float64, 1e-11), (wp.float32, 1e-5)])
@pytest.mark.parametrize("active_widths", [False, True])
def test_captured_cg_matches_changing_active_rhs_blocks(dtype, tolerance, active_widths):
    from benchmarks.experimental.batched_cg import CapturedActiveCG, CapturedPaddedCG

    a = sp.diags([-np.ones(36), np.full(37, 4.0), -np.ones(36)], [-1, 0, 1], format="csr")
    inverse = BatchedCGInverse(matrix(a), 5, tol=tolerance * 0.1, maxiter=30, working_dtype=dtype)
    adapter = CapturedActiveCG(inverse) if active_widths else CapturedPaddedCG(inverse)
    if active_widths:
        adapter.reset_stats()
        for item in adapter.adapters.values():
            assert item.inverse.operator.matrix.values.ptr == inverse.operator.matrix.values.ptr
            assert item.inverse.operator.matrix.offsets.ptr == inverse.operator.matrix.offsets.ptr
    else:
        inverse.reset_stats()
    rng = np.random.default_rng(840)
    original_values = inverse.source_matrix.values.numpy()
    for i, (width, scale) in enumerate([(5, 1.0), (1, 1e-4), (3, 1e4), (2, 0.0), (5, 1.0)]):
        matrix_scale = [1.0, 1.3, 0.7, 2.0, 1.05][i]
        inverse.source_matrix.values.assign(original_values * matrix_scale)
        source = rng.normal(size=(width, 37)) * scale
        rhs = wp.array(source, dtype=wp.float64)
        # In-place input/output must not overwrite input before the graph sees it.
        adapter.apply(rhs, rhs)
        actual = rhs.numpy()
        residual = np.linalg.norm(matrix_scale * (a @ actual.T).T - source, axis=1)
        assert np.all(residual <= tolerance * np.linalg.norm(source, axis=1))
        assert np.isfinite(actual).all()
    stats = adapter.collect_stats().numpy() if active_widths else inverse.stats.numpy()
    assert stats[2] == 0
    assert stats[3] == (16 if active_widths else 25)


@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("dtype,tolerance", [(wp.float64, 1e-11), (wp.float32, 1e-5)])
@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_batched_cg_changed_source_values_replay(layout, block, dtype, tolerance, tmp_path):
    a, wa = fixture(block)
    pre = linear.FSAI(wa, max_row_size=8, factor_dtype=wp.float32)
    inverse = BatchedCGInverse(
        wa,
        3,
        tol=tolerance * 0.1,
        maxiter=50,
        preconditioner=pre,
        working_dtype=dtype,
        fsai_layout=layout,
        matrix_layout=layout,
    )
    rhs = wp.zeros((3, len(a)), dtype=wp.float64)
    inverse.apply(rhs, rhs)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        inverse.reset_stats()
        inverse.apply(rhs, rhs)
    rng = np.random.default_rng(51)
    for spread in (0.0, 0.4, -0.2):
        diagonal = np.linspace(1, 1 + spread, len(a))
        current = diagonal[:, None] * a * diagonal[None, :]
        host = sp.csr_matrix(current)
        if block == 3:
            host = host.tobsr(blocksize=(3, 3))
        np.testing.assert_array_equal(wa.columns.numpy()[: len(host.indices)], host.indices)
        values = np.zeros_like(wa.values.numpy())
        values[: len(host.data)] = host.data
        wa.values.assign(values)
        source = rng.normal(size=rhs.shape)
        rhs.assign(source)
        wp.capture_launch(capture.graph)
        actual = rhs.numpy()
        residual = np.linalg.norm(actual @ current.T - source, axis=1)
        assert np.all(residual <= tolerance * np.linalg.norm(source, axis=1))
        expected = np.linalg.solve(current, source.T).T
        assert np.all(
            np.linalg.norm(actual - expected, axis=1)
            <= tolerance * np.linalg.norm(expected, axis=1)
        )
        assert inverse.stats.numpy()[2:].tolist() == [0, 3]
    dot = tmp_path / "changed-source.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text
