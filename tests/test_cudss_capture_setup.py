"""Fast capture preparation must not depend on a preceding eigen solve."""

import numpy as np
import pytest
import warp as wp

from benchmarks.graph_audit import graph_summary
from tests.test_solver import sparse
from warpack import CuDSSInverse, KrylovSchur


@pytest.mark.parametrize("source", ["x", "b"])
def test_capture_preserves_start_in_inverse_workspace(source):
    inverse = CuDSSInverse(sparse(np.diag(np.arange(1.0, 49.0))), 1)
    start = getattr(inverse, source)
    values = np.random.default_rng(17).normal(size=(1, 48))
    wp.copy(start, wp.array(values, dtype=wp.float64))
    solver = KrylovSchur(inverse, 3, ncv=32, which="LA", tol=1e-10, initial_vector=start)
    try:
        graph = solver.capture(3, adaptive=False)
        np.testing.assert_array_equal(start.numpy(), values)
        for _ in range(2):
            wp.capture_launch(graph)
            assert solver.converged.numpy()[0] == 3
            np.testing.assert_allclose(1 / solver.eigenvalues.numpy(), [1, 2, 3], rtol=1e-9)
            np.testing.assert_array_equal(start.numpy(), values)
    finally:
        inverse.close()


@pytest.mark.parametrize(
    "mtype,matching", [("symmetric", False), ("general", False), ("general", True)]
)
def test_indefinite_factor_first_capture(mtype, matching):
    # CuDSS storage/pivoting options must not require a nonzero eager solve.
    n = 64
    a = np.diag(np.arange(n, dtype=float) - 31.5) + 0.1 * (np.eye(n, k=1) + np.eye(n, k=-1))
    spectrum = np.linalg.eigvalsh(a)
    expected = spectrum[spectrum > 0][:4]
    inverse = CuDSSInverse(sparse(a), 1, mtype=mtype, matching=matching)
    solver = KrylovSchur(inverse, 4, ncv=32, which="LA", tol=1e-11)
    apply = inverse.apply
    eager_calls = []

    def counted(x, y):
        if not wp.get_stream().is_capturing:
            eager_calls.append(1)
        apply(x, y)

    inverse.apply = counted
    try:
        graph = solver.capture(4, adaptive=False)
        assert len(eager_calls) == 1
        for _ in range(2):
            wp.capture_launch(graph)
            assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == 4
            lam = 1 / solver.eigenvalues.numpy()
            v = solver.eigenvectors.numpy().T
            np.testing.assert_allclose(lam, expected, rtol=1e-10, atol=1e-10)
            assert np.linalg.norm(a @ v - v * lam, axis=0).max() < 1e-8
    finally:
        inverse.close()


@pytest.mark.parametrize("ncv", [32, 64])
@pytest.mark.parametrize("refinement_steps", [0, 1])
def test_first_capture_primes_once_and_replays_changed_starts(tmp_path, ncv, refinement_steps):
    n, k = 96, 4
    a = np.diag(np.arange(1.0, n + 1)) + 0.15 * (np.eye(n, k=1) + np.eye(n, k=-1))
    reference = np.linalg.eigvalsh(a)[:k]
    inverse = CuDSSInverse(sparse(a), 1, refinement_steps=refinement_steps)
    start = wp.array(np.random.default_rng(7).normal(size=(1, n)), dtype=wp.float64)
    solver = KrylovSchur(inverse, k, ncv=ncv, which="LA", tol=1e-11, initial_vector=start)
    apply = inverse.apply
    eager_calls = []

    def counted(x, y):
        if not wp.get_stream().is_capturing:
            eager_calls.append(1)
        apply(x, y)

    inverse.apply = counted
    try:
        graph = solver.capture(3, adaptive=False)
        assert len(eager_calls) == 1
        fast_dot = str(tmp_path / "fast.dot")
        wp.capture_debug_dot_print(graph, fast_dot)
        fast_nodes = graph_summary(fast_dot)["node_types_including_conditional_body"]
        assert not fast_nodes.get("HOST", 0)
        for seed in [11, 42, 11]:
            values = np.random.default_rng(seed).normal(size=(1, n))
            wp.copy(start, wp.array(values, dtype=wp.float64))
            wp.capture_launch(graph)
            assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
            v = solver.eigenvectors.numpy().T
            lam = 1 / solver.eigenvalues.numpy()
            np.testing.assert_allclose(lam, reference, rtol=1e-10, atol=1e-10)
            assert np.linalg.norm(a @ v - v * lam, axis=0).max() < 1e-8
            np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-10)
            np.testing.assert_array_equal(start.numpy(), values)

        # Compare the actual recorded program against the legacy full warmup.
        from warpack import SymmetricEigensolver

        solver._prepare_capture = lambda finalizer: SymmetricEigensolver._prepare_capture(
            solver, finalizer
        )
        control = solver.capture(3, adaptive=False)
        control_dot = str(tmp_path / "control.dot")
        wp.capture_debug_dot_print(control, control_dot)
        assert graph_summary(control_dot)["node_types_including_conditional_body"] == fast_nodes
        wp.capture_launch(control)
        assert solver.converged.numpy()[0] == k
        np.testing.assert_allclose(1 / solver.eigenvalues.numpy(), reference, rtol=1e-10)
    finally:
        inverse.close()


@pytest.mark.parametrize("fallback", ["driver", "finalizer", "operator", "solver"])
def test_unknown_capture_preparation_keeps_full_warmup(monkeypatch, fallback):
    class CustomInverse(CuDSSInverse):
        pass

    class CustomSolver(KrylovSchur):
        pass

    inverse_type = CustomInverse if fallback == "operator" else CuDSSInverse
    solver_type = CustomSolver if fallback == "solver" else KrylovSchur
    inverse = inverse_type(sparse(np.diag(np.arange(1.0, 41.0))), 1)
    solver = solver_type(inverse, 3, ncv=24, which="LA", tol=1e-10)
    if fallback == "driver":
        monkeypatch.setattr(wp, "get_cuda_driver_version", lambda: (12, 2))
    calls = []
    apply = inverse.apply

    def counted(x, y):
        if not wp.get_stream().is_capturing:
            calls.append(1)
        apply(x, y)

    inverse.apply = counted
    output = wp.empty_like(solver.eigenvalues)
    finalizer_warmups = []

    def finalizer():
        if not wp.get_stream().is_capturing:
            assert solver.converged.numpy()[0] == 3
            finalizer_warmups.append(1)
        wp.copy(output, solver.eigenvalues)

    try:
        graph = solver.capture(
            3, adaptive=False, finalizer=finalizer if fallback == "finalizer" else None
        )
        assert len(calls) >= 24
        wp.capture_launch(graph)
        assert solver.converged.numpy()[0] == 3
        np.testing.assert_allclose(1 / solver.eigenvalues.numpy(), [1, 2, 3], rtol=1e-9)
        if fallback == "finalizer":
            assert len(finalizer_warmups) == 1
            np.testing.assert_array_equal(output.numpy(), solver.eigenvalues.numpy())
    finally:
        inverse.close()
