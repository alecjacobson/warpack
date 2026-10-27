"""A converged guard must not starve an unconverged requested eigenvector."""

import numpy as np
import pytest
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import complex_matrix, output
from warpack import GeneralEigensolver


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("nonnormal", [False, True])
def test_converged_guard_recovers_unfinished_target(scale, nonnormal, tmp_path):
    n, k, m = 96, 4, 16
    rng = np.random.default_rng(281)
    unitary = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    values = np.arange(1.0, n + 1) + 0.1j * np.sin(np.arange(n))
    triangular = np.diag(values)
    if nonnormal:
        triangular += np.diag(np.full(n - 1, 0.02 + 0.01j), 1)
    a = scale * (unitary @ triangular @ unitary.conj().T)
    solver = GeneralEigensolver(complex_matrix(a), k, ncv=m, which="SR", tol=1e-12)
    initial = wp.zeros_like(solver.q)

    def initialize(seed=42):
        solver.status.zero_()
        solver.restart_index.fill_(-1)
        solver.residual_restarts.zero_()
        wp.copy(solver.q, initial)

    solver.initialize = initialize

    def set_start(angle):
        basis = np.zeros((m, n), dtype=complex)
        basis[: solver.keep] = unitary[:, : solver.keep].T
        basis[0] = (unitary[:, 0] + angle * unitary[:, -1]) / np.sqrt(1 + angle**2)
        initial.assign(np.stack([basis.real, basis.imag], axis=-1))

    set_start(1e-9)
    graph = solver.capture(20)
    for angle in (1e-9, -2e-9):
        set_start(angle)
        wp.capture_launch(graph)
        w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
        assert solver.residual_restarts.numpy()[0] > 0
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == k, solver.residuals.numpy()
        np.testing.assert_allclose(w, scale * values[:k], rtol=2e-12, atol=2e-12)
        assert np.max(np.linalg.norm(a @ v - v * w, axis=0) / np.maximum(1, abs(w))) < 1e-12
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
    dot = tmp_path / "general-recovery.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


@pytest.mark.parametrize("candidate", ["in_span", "nonfinite", "independent"])
def test_unusable_residual_keeps_independent_completion(candidate):
    n = 32
    solver = GeneralEigensolver(complex_matrix(np.diag(np.arange(1.0, n + 1))), 3, ncv=12)
    basis = np.zeros((solver.m, n, 2))
    for j in range(solver.keep):
        basis[j, j, 0] = 1
    initial = wp.array(basis, dtype=wp.vec2d)
    seed = np.zeros((n, 2))
    seed[0 if candidate == "in_span" else -1, 0] = 1
    if candidate == "nonfinite":
        seed[:] = np.nan
    solver.restart_residual.assign(seed)

    def expand():
        wp.copy(solver.q, initial)
        solver.restart_index.fill_(0)
        solver.residual_restarts.zero_()
        solver.residual_fallbacks.zero_()
        solver.expand(solver.keep)

    expand()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        expand()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        q = output(solver.q)
        row = q[solver.keep]
        assert solver.residual_restarts.numpy()[0] == 1
        assert solver.residual_fallbacks.numpy()[0] == (candidate != "independent")
        np.testing.assert_allclose(np.linalg.norm(row), 1, atol=1e-12)
        np.testing.assert_allclose(q[: solver.keep].conj() @ row, 0, atol=1e-12)


@pytest.mark.parametrize("which", ["LM", "SM", "LR", "SR", "LI", "SI"])
def test_completed_nonnormal_space_preserves_every_selector(which):
    # Full-space replay deliberately continues after exact convergence. Tiny
    # cached residuals must not destroy the requested spectrum or add zero rows.
    n, k = 24, 4
    rng = np.random.default_rng(837)
    values = rng.normal(size=n) + 1j * rng.normal(size=n)
    triangular = np.diag(values) + np.diag(np.full(n - 1, 0.05 + 0.1j), 1)
    unitary = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    a = unitary @ triangular @ unitary.conj().T
    key = {
        "LM": -abs(values),
        "SM": abs(values),
        "LR": -values.real,
        "SR": values.real,
        "LI": -values.imag,
        "SI": values.imag,
    }[which]
    expected = values[np.argsort(key)[:k]]
    solver = GeneralEigensolver(complex_matrix(a), k, ncv=n, which=which, tol=1e-12)
    graph = solver.capture(3, adaptive=False)
    for _ in range(2):
        wp.capture_launch(graph)
        w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        cost = abs(w[:, None] - expected)
        rows, cols = linear_sum_assignment(cost)
        assert cost[rows, cols].max() < 1e-11
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
        assert np.max(np.linalg.norm(a @ v - v * w, axis=0) / np.maximum(1, abs(w))) < 1e-12
