"""Invalid budgets must fail before warmup or mutation, including wrappers."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import complex_matrix, output
from warpack import GeneralEigensolver, ShiftInvertEigensolver


def make_solver(shifted):
    a = np.diag(np.arange(1.0, 9.0)).astype(complex)
    a += np.diag(np.full(7, 0.2 + 0.1j), 1)
    op = complex_matrix(a)
    solver = (
        ShiftInvertEigensolver(op, 2, sigma=0.5, ncv=6, tol=1e-12)
        if shifted
        else GeneralEigensolver(op, 2, ncv=6, tol=1e-12)
    )
    return a, solver


def close(solver):
    solver.graph = None
    if hasattr(solver, "core"):
        solver.core.graph = None
        solver.close()


@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("method", ["solve", "capture"])
@pytest.mark.parametrize("budget,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_invalid_budget_precedes_initialization(shifted, method, budget, error, monkeypatch):
    _, solver = make_solver(shifted)
    core = getattr(solver, "core", solver)

    def initialize(*args, **kwargs):
        raise AssertionError("invalid budget reached initialization")

    monkeypatch.setattr(core, "initialize", initialize)
    core.converged.fill_(-1)
    core.eigenvectors.fill_(wp.vec2d(3.0, 4.0))
    try:
        with pytest.raises(error):
            getattr(solver, method)(budget)
        assert core.converged.numpy()[0] == -1
        np.testing.assert_array_equal(core.eigenvectors.numpy(), np.tile([3.0, 4.0], (2, 8, 1)))
    finally:
        close(solver)


@pytest.mark.parametrize("shifted", [False, True])
def test_integer_budget_and_short_replay_keep_valid_vectors(shifted):
    a, solver = make_solver(shifted)
    graph = None
    try:
        solver.solve(np.int64(1))
        graph = solver.capture(np.int64(1))
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
            np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
            residuals = np.linalg.norm(a @ v - v * w, axis=0) / np.maximum(1, abs(w))
            assert np.all(np.isfinite(residuals))
            assert solver.converged.numpy()[0] == np.count_nonzero(residuals <= 1e-12)
            assert solver.converged.numpy()[0] < 2
    finally:
        graph = None
        close(solver)


@pytest.mark.parametrize("shifted", [False, True])
def test_legitimate_zero_modes_have_nonzero_vectors(shifted):
    op = complex_matrix(np.zeros((8, 8)))
    solver = (
        ShiftInvertEigensolver(op, 2, sigma=0.5, ncv=6, tol=1e-12)
        if shifted
        else GeneralEigensolver(op, 2, ncv=6, tol=1e-12)
    )
    graph = None
    try:
        graph = solver.capture(2)
        wp.capture_launch(graph)
        v = output(solver.eigenvectors).T
        assert solver.converged.numpy()[0] == 2
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(output(solver.eigenvalues), 0, atol=1e-12)
    finally:
        graph = None
        close(solver)


@pytest.mark.parametrize("shifted", [False, True])
def test_ordinary_solve_updates_iterations_after_graph_replay(shifted):
    _, solver = make_solver(shifted)
    core = getattr(solver, "core", solver)
    try:
        solver.solve(2)
        assert core.iterations.numpy()[0] == 2
        graph = solver.capture(3) if shifted else solver.capture(3, adaptive=False)
        wp.capture_launch(graph)
        assert core.iterations.numpy()[0] == 3
        solver.solve(1)
        assert core.iterations.numpy()[0] == 1
    finally:
        close(solver)
