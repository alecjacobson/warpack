"""CG in-place RHS preservation and optional true-linear-residual diagnostics."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import CGInverse


def capture(function):
    function()
    wp.synchronize_device()
    with wp.ScopedCapture() as captured:
        function()
    return captured.graph


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_changed_scaled_rhs_and_device_audit(audit, inplace, tmp_path):
    n, width = 129, 3
    a = sp.diags([-np.ones(n - 1), np.full(n, 4.0), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    inverse = CGInverse(
        matrix(a), tol=1e-11, atol=0.0, maxiter=1000, preconditioner="diag", verify_residuals=audit
    )
    inputs = wp.zeros((width, n), dtype=wp.float64)
    output = wp.empty_like(inputs)

    def body():
        inverse.reset_stats()
        if inplace:
            wp.copy(output, inputs)
        inverse.apply(output if inplace else inputs, output)

    graph = capture(body)
    rng = np.random.default_rng(194)
    for scale in [1.0, 1e-6, 1e6, 1.0]:
        rhs = scale * rng.normal(size=(width, n))
        rhs[-1] = 0.0
        inputs.assign(rhs)
        wp.capture_launch(graph)
        result = output.numpy()
        expected = sla.spsolve(a, rhs.T).T
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-10 * scale)
        np.testing.assert_array_equal(inputs.numpy(), rhs)
        stats = inverse.stats.numpy()
        assert stats[2] == 0 and stats[3] == width
        if audit:
            fresh = inverse.fresh_stats.numpy()
            assert fresh[2] == 0 and fresh[3] == width
            np.testing.assert_array_equal(fresh[[0, 1, 3]], stats[[0, 1, 3]])
            residual = np.linalg.norm(a @ result.T - rhs.T, axis=0)
            norms = np.linalg.norm(rhs, axis=1)
            relative = np.divide(residual, norms, out=np.zeros_like(residual), where=norms > 0)
            maximum = inverse.residual_maxima.numpy()
            np.testing.assert_allclose(
                maximum[0], residual.max(), rtol=0.01, atol=1e-13 * max(1, scale)
            )
            np.testing.assert_allclose(maximum[1], relative.max(), rtol=0.01, atol=1e-13)
        else:
            assert inverse.fresh_stats is None and inverse.residual_maxima is None
    dot = tmp_path / "cg-audit.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    info = graph_summary(dot)
    assert info["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in info["memcpy_kinds"])
    assert ("residual_norms" in dot.read_text()) == audit


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("workspace", ["b", "x"])
def test_exact_alias_of_exposed_workspace(audit, workspace):
    n = 32
    diagonal = np.linspace(1.0, 4.0, n)
    inverse = CGInverse(
        matrix(sp.diags(diagonal, format="csr")),
        tol=1e-12,
        preconditioner="diag",
        verify_residuals=audit,
    )
    values = getattr(inverse, workspace).reshape((1, n))
    rhs = np.arange(1.0, n + 1).reshape(1, n)
    values.assign(rhs)
    inverse.apply(values, values)
    np.testing.assert_allclose(values.numpy(), rhs / diagonal, rtol=1e-12)
    if audit:
        assert inverse.fresh_stats.numpy()[2] == 0


def test_partial_overlap_rejected_before_clobbering_rhs():
    n = 16
    inverse = CGInverse(matrix(sp.eye(n, format="csr")))
    source = np.arange(3 * n, dtype=float).reshape(3, n)
    storage = wp.array(source, dtype=wp.float64)
    with pytest.raises(ValueError, match="partially overlapping"):
        inverse.apply(storage[:2], storage[1:])
    np.testing.assert_array_equal(storage.numpy(), source)


def test_recursive_convergence_does_not_hide_fresh_residual_failure():
    n = 257
    a = sp.diags([-np.ones(n - 1), np.full(n, 2.0), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    inverse = CGInverse(matrix(a), tol=1e-14, atol=0.0, maxiter=2000, verify_residuals=True)
    rhs = np.random.default_rng(918).normal(size=(1, n))
    inputs = wp.array(rhs, dtype=wp.float64)
    output = wp.empty_like(inputs)

    def body():
        inverse.reset_stats()
        inverse.apply(inputs, output)

    graph = capture(body)
    for _ in range(2):
        wp.capture_launch(graph)
        result = output.numpy()
        relative = np.linalg.norm(a @ result[0] - rhs[0]) / np.linalg.norm(rhs)
        assert inverse.stats.numpy()[2] == 0
        assert inverse.fresh_stats.numpy()[2] == 1
        assert relative > 5e-14
        assert inverse.residual_maxima.numpy()[1] > 5e-14
        np.testing.assert_allclose(result[0], sla.spsolve(a, rhs[0]), rtol=1e-9, atol=1e-9)


def test_audit_nonfinite_failure_and_recovery_on_same_graph():
    n = 16
    inverse = CGInverse(matrix(sp.eye(n, format="csr")), verify_residuals=True)
    inputs = wp.ones((1, n), dtype=wp.float64)
    output = wp.empty_like(inputs)

    def body():
        inverse.reset_stats()
        inverse.apply(inputs, output)

    graph = capture(body)
    bad = np.ones((1, n))
    bad[0, 3] = np.nan
    inputs.assign(bad)
    wp.capture_launch(graph)
    assert inverse.fresh_stats.numpy()[2] == 1
    assert np.isinf(inverse.residual_maxima.numpy()).all()
    inputs.fill_(1.0)
    wp.capture_launch(graph)
    assert inverse.fresh_stats.numpy()[2] == 0
    np.testing.assert_allclose(output.numpy(), 1.0, atol=1e-12)


def test_audit_uses_current_matrix_values_on_replay():
    n = 32
    diagonal = np.linspace(1.0, 2.0, n)
    a = matrix(sp.diags(diagonal, format="csr"))
    a.nnz_sync()
    inverse = CGInverse(a, tol=1e-11, verify_residuals=True)
    rhs = np.random.default_rng(99).normal(size=(1, n))
    inputs = wp.array(rhs, dtype=wp.float64)
    output = wp.empty_like(inputs)

    def body():
        inverse.reset_stats()
        inverse.apply(inputs, output)

    graph = capture(body)
    for scale in [1.0, 2.0, 0.5, 1.0]:
        a.values[:n].assign(scale * diagonal)
        wp.capture_launch(graph)
        np.testing.assert_allclose(output.numpy(), rhs / (scale * diagonal), rtol=1e-9, atol=1e-10)
        assert inverse.fresh_stats.numpy()[2] == 0


@pytest.mark.parametrize("block_size", [2, 3])
def test_audit_supports_upstream_bsr_block_shapes(block_size):
    import warp.sparse as ws

    count = 7
    scalar = sp.diags([-np.ones(count - 1), np.full(count, 4.0), -np.ones(count - 1)], [-1, 0, 1])
    block = np.eye(block_size) + np.full((block_size, block_size), 0.1)
    a = sp.kron(scalar, block).tobsr(blocksize=(block_size, block_size))
    dtype = wp.types.matrix(shape=(block_size, block_size), dtype=wp.float64)
    device = ws.bsr_from_triplets(
        count,
        count,
        wp.array(np.repeat(np.arange(count), np.diff(a.indptr)), dtype=wp.int32),
        wp.array(a.indices, dtype=wp.int32),
        wp.array(a.data, dtype=dtype),
    )
    inverse = CGInverse(device, tol=1e-12, atol=0.0, verify_residuals=True)
    rhs = np.random.default_rng(72).normal(size=(2, a.shape[0]))
    inputs = wp.array(rhs, dtype=wp.float64)
    output = wp.empty_like(inputs)

    def body():
        inverse.reset_stats()
        inverse.apply(inputs, output)

    graph = capture(body)
    for _ in range(2):
        wp.capture_launch(graph)
        np.testing.assert_allclose(
            output.numpy(), sla.spsolve(a.tocsr(), rhs.T).T, rtol=1e-10, atol=1e-11
        )
        assert inverse.fresh_stats.numpy()[2] == 0
