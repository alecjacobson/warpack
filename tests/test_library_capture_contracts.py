"""Public setup must fail before mutation and leave caller capture usable."""

import numpy as np
import pytest
import warp as wp

from tests.test_solver import sparse
from warpack import BlockLOBPCG, KrylovSchur, SparseOperator, SymmetricEigensolver
from warpack.arnoldi import ComplexCSR, GeneralEigensolver


def make_solver(kind):
    a = sparse(np.diag(np.arange(1.0, 17.0)))
    op = SparseOperator(a)
    if kind == "symmetric":
        return SymmetricEigensolver(op, 2, ncv=16)
    if kind == "krylov":
        return KrylovSchur(op, 2, ncv=16, which="LA")
    if kind == "block":
        return BlockLOBPCG(op, 2, width=4)
    values = wp.array(
        a.values.numpy().astype(complex).view(np.float64).reshape(-1, 2), dtype=wp.vec2d
    )
    return GeneralEigensolver(ComplexCSR(16, a.offsets, a.columns, values), 2, ncv=12)


@pytest.mark.parametrize("kind", ["symmetric", "krylov"])
@pytest.mark.parametrize("method", ["solve", "capture"])
@pytest.mark.parametrize("budget", [1.5, np.float64(2.0)])
def test_fractional_budget_fails_before_warmup(kind, method, budget, monkeypatch):
    solver = make_solver(kind)
    solver.eigenvalues.fill_(123.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid budget mutated solver state")

    monkeypatch.setattr(solver, "initialize", forbidden)
    with pytest.raises(TypeError):
        getattr(solver, method)(budget)
    np.testing.assert_array_equal(solver.eigenvalues.numpy(), [123.0, 123.0])


@pytest.mark.parametrize("kind", ["symmetric", "krylov"])
def test_eager_integer_budget_replaces_replay_count(kind):
    solver = make_solver(kind)
    graph = solver.capture(np.int64(3), adaptive=False)
    wp.capture_launch(graph)
    assert solver.iterations.numpy()[0] == 3
    solver.solve(np.int64(1))
    assert solver.iterations.numpy()[0] == 1
    w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
    expected = [1.0, 2.0] if kind == "symmetric" else [16.0, 15.0]
    np.testing.assert_allclose(w, expected, atol=1e-5)
    assert np.linalg.norm(np.arange(1.0, 17.0)[:, None] * v - v * w, axis=0).max() < 1e-2
    np.testing.assert_allclose(v.T @ v, np.eye(2), atol=1e-10)


@pytest.mark.parametrize("kind", ["symmetric", "krylov", "block", "general"])
@pytest.mark.parametrize("other_stream", [False, True])
def test_nested_standalone_capture_rejected_without_poisoning_outer(
    kind, other_stream, monkeypatch
):
    solver = make_solver(kind)
    solver.solve(1)
    source = wp.array([7.0], dtype=wp.float64)
    output = wp.zeros_like(source)
    stream = wp.Stream(solver.device) if other_stream else wp.get_stream(solver.device)
    wp.synchronize_device(solver.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("nested capture reached warmup")

    monkeypatch.setattr(solver, "initialize", forbidden)
    with wp.ScopedCapture(stream=stream) as capture:
        with pytest.raises(RuntimeError, match="already capturing"):
            solver.capture(2)
        wp.copy(output, source, stream=stream)
    wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    with wp.ScopedStream(stream):
        for value in [9.0, 13.0]:
            source.fill_(value)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(output.numpy(), [value])
