"""Fresh returned-value checks complement cached Krylov residual estimates."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import CGInverse, CuDSSInverse, EigenpairEvaluation, SparseOperator


def capture(evaluation):
    evaluation.run()
    wp.synchronize_device()
    with wp.ScopedCapture() as cap:
        evaluation.run()
    return cap.graph


def test_returned_values_are_checked_without_rayleigh_replacement(tmp_path):
    n, k = 17, 3
    diagonal = np.arange(1.0, n + 1)
    device = matrix(sp.diags(diagonal, format="csr"))
    vectors = wp.array(np.eye(n)[:k], dtype=wp.float64)
    values = wp.array(diagonal[:k], dtype=wp.float64)
    evaluation = EigenpairEvaluation(SparseOperator(device), vectors, eigenvalues=values, tol=1e-10)
    graph = capture(evaluation)
    for scale, delta in [(1.0, 0.0), (2.0, 0.5), (0.25, 0.0)]:
        device.values[:n].assign(scale * diagonal)
        supplied = scale * diagonal[:k]
        supplied[1] += delta
        values.assign(supplied)
        wp.capture_launch(graph)
        expected = abs(scale * diagonal[:k] - supplied) / np.maximum(1, abs(supplied))
        np.testing.assert_allclose(evaluation.residuals.numpy(), expected, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(evaluation.values.numpy(), supplied)
        np.testing.assert_array_equal(values.numpy(), supplied)
        assert evaluation.converged.numpy()[0] == np.count_nonzero(expected <= 1e-10)
    # Default behavior still reports fresh Rayleigh values.
    rayleigh = EigenpairEvaluation(SparseOperator(device), vectors, tol=1e-10)
    rayleigh.run()
    np.testing.assert_allclose(rayleigh.values.numpy(), 0.25 * diagonal[:k])
    assert rayleigh.converged.numpy()[0] == k
    dot = tmp_path / "fresh-values.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


@pytest.mark.parametrize("backend", ["cg", "cudss"])
@pytest.mark.parametrize("width", [1, 2])
def test_fixed_rhs_batches_and_changed_vectors(backend, width, tmp_path):
    n, k = 23, 4
    a = sp.diags([-np.ones(n - 1), np.full(n, 4.0), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    inverse = (
        CGInverse(matrix(a), tol=1e-13, atol=0.0)
        if backend == "cg"
        else CuDSSInverse(matrix(a), width)
    )
    vectors = wp.zeros((k, n), dtype=wp.float64)
    supplied = wp.zeros(k, dtype=wp.float64)
    evaluation = EigenpairEvaluation(inverse, vectors, eigenvalues=supplied, rhs_width=width)
    graph = capture(evaluation)
    rng = np.random.default_rng(153)
    try:
        for _ in range(2):
            x = rng.normal(size=(k, n))
            x /= np.linalg.norm(x, axis=1)[:, None]
            mu = rng.normal(size=k)
            vectors.assign(x)
            supplied.assign(mu)
            wp.capture_launch(graph)
            ax = np.linalg.solve(a.toarray(), x.T).T
            expected = np.linalg.norm(ax - mu[:, None] * x, axis=1) / np.maximum(1, abs(mu))
            np.testing.assert_allclose(
                evaluation.residuals.numpy(), expected, rtol=1e-11, atol=1e-13
            )
            np.testing.assert_allclose(evaluation.ax.numpy(), ax, rtol=1e-10, atol=1e-13)
            np.testing.assert_array_equal(vectors.numpy(), x)
            assert evaluation.converged.numpy()[0] == 0
        dot = tmp_path / "batched-check.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
        assert not any("DtoH" in kind for kind in audit["memcpy_kinds"])
        if backend == "cg":
            assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
    finally:
        graph = None
        if backend == "cudss":
            inverse.close()


def test_physical_accuracy_does_not_imply_tight_inverse_accuracy():
    n, sigma = 20, -1e-6
    eye = np.eye(n)
    a = eye - 0.5 * np.roll(eye, 1, axis=0) - 0.5 * np.roll(eye, -1, axis=0)
    lam = 2 * np.sin(np.pi / n) ** 2
    # A tiny known rigid component is amplified by the near-null inverse.
    v = np.sqrt(2 / n) * np.cos(2 * np.pi * np.arange(n) / n) + 1e-10 / np.sqrt(n)
    v /= np.linalg.norm(v)
    vectors = wp.array(v.reshape(1, n), dtype=wp.float64)
    inverse = CuDSSInverse(matrix(a - sigma * eye), 1, mtype="symmetric")
    transformed = wp.array([1 / (lam - sigma)], dtype=wp.float64)
    physical_values = wp.array([lam], dtype=wp.float64)
    fresh = EigenpairEvaluation(inverse, vectors, eigenvalues=transformed, rhs_width=1, tol=1e-12)
    physical = EigenpairEvaluation(SparseOperator(matrix(a)), vectors, eigenvalues=physical_values)
    graph = capture(fresh)
    try:
        wp.capture_launch(graph)
        physical.run()
        assert physical.converged.numpy()[0] == 1
        assert physical.residuals.numpy()[0] < 1e-10
        assert fresh.converged.numpy()[0] == 0
        assert fresh.residuals.numpy()[0] > 1e-6
    finally:
        graph = None
        inverse.close()


def test_invalid_values_and_vectors_recover_on_same_graph():
    n, k = 8, 2
    vectors = wp.array(np.eye(n)[:k], dtype=wp.float64)
    values = wp.ones(k, dtype=wp.float64)
    evaluation = EigenpairEvaluation(SparseOperator(matrix(sp.eye(n))), vectors, eigenvalues=values)
    graph = capture(evaluation)
    values.assign([np.nan, np.inf])
    wp.capture_launch(graph)
    assert evaluation.converged.numpy()[0] == 0
    values.fill_(1.0)
    bad = np.eye(n)[:k]
    bad[0, 3] = np.nan
    vectors.assign(bad)
    wp.capture_launch(graph)
    assert evaluation.converged.numpy()[0] == 1
    vectors.assign(np.eye(n)[:k])
    wp.capture_launch(graph)
    assert evaluation.converged.numpy()[0] == k
    np.testing.assert_array_equal(evaluation.residuals.numpy(), np.zeros(k))


@pytest.mark.parametrize("width", [0, -1, 3])
def test_invalid_batch_width(width):
    op = SparseOperator(matrix(sp.eye(8)))
    with pytest.raises(ValueError, match="rhs_width"):
        EigenpairEvaluation(op, wp.zeros((4, 8), dtype=wp.float64), rhs_width=width)


@pytest.mark.parametrize("dtype,shape", [(wp.float32, (2,)), (wp.float64, (2, 1))])
def test_invalid_eigenvalues(dtype, shape):
    op = SparseOperator(matrix(sp.eye(8)))
    with pytest.raises(ValueError, match="eigenvalues"):
        EigenpairEvaluation(
            op, wp.zeros((2, 8), dtype=wp.float64), eigenvalues=wp.zeros(shape, dtype=dtype)
        )
