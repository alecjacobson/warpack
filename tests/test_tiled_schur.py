"""Independent Schur similarity, eigenpairs, changed replay and bounded failure."""

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import output
from warpack.schur_tiled import TiledSchur


@pytest.fixture(params=["tiled", "fused"])
def solver_type(request):
    if request.param == "fused":
        from benchmarks.experimental.schur_fused import FusedSchur

        return FusedSchur
    return TiledSchur


def workspace(a, solver_type, budget=None):
    data = np.stack([a.real, a.imag], axis=-1)
    source = wp.array(data, dtype=wp.vec2d)
    h, q, v = wp.empty_like(source), wp.empty_like(source), wp.empty_like(source)
    n = len(a)
    values, u, ss = (wp.empty(n, dtype=wp.vec2d) for _ in range(3))
    cs = wp.empty(n, dtype=wp.float64)
    status = wp.zeros(1, dtype=wp.int32)
    solver = solver_type(h, q, v, values, u, cs, ss, status, maxiter=budget)
    return source, solver


@pytest.mark.parametrize("n", [1, 4, 17, 32, 50, 64])
@pytest.mark.parametrize("complex_input", [False, True])
def test_changed_projection(n, complex_input, tmp_path, solver_type):
    rng = np.random.default_rng(81)
    a = rng.normal(size=(n, n)).astype(complex)
    if complex_input:
        a += 1j * rng.normal(size=(n, n))
    source, solver = workspace(a, solver_type)
    wp.copy(solver.h, source)
    solver.run()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.copy(solver.h, source)
        solver.run()
    for scale in (1e-6, 1.0, 1e6):
        matrix = scale * (a + 0.01 * rng.normal(size=(n, n)))
        source.assign(np.stack([matrix.real, matrix.imag], axis=-1))
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == 0
        h, q, v, w = (output(x) for x in (solver.h, solver.q, solver.vectors, solver.values))
        reference = la.eigvals(matrix)
        norm = max(np.linalg.norm(matrix), np.finfo(float).tiny)
        assert np.linalg.norm(matrix @ q - q @ h) / norm < 1e-11
        assert np.linalg.norm(q.conj().T @ q - np.eye(n)) < 1e-11
        assert np.linalg.norm(np.tril(h, -1)) / norm < 1e-11
        assert np.max(np.linalg.norm(matrix @ v - v * w, axis=0)) / norm < 1e-11
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
        cost = abs(w[:, None] - reference)
        rows, cols = linear_sum_assignment(cost)
        assert cost[rows, cols].max() / norm < 1e-10
    dot = tmp_path / "schur.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


def test_qr_iteration_budget_reports_failure(solver_type):
    a = np.random.default_rng(91).normal(size=(32, 32)).astype(complex)
    source, solver = workspace(a, solver_type, budget=1)
    wp.copy(solver.h, source)
    solver.run()
    assert solver.status.numpy()[0] == 2
    assert solver.state.numpy()[1] == 1


def test_nonfinite_projection_then_same_graph_recovers(solver_type):
    a = np.random.default_rng(92).normal(size=(17, 17)).astype(complex)
    source, solver = workspace(a, solver_type)
    wp.copy(solver.h, source)
    solver.run()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.copy(solver.h, source)
        solver.run()
    source.fill_(wp.vec2d(float("nan"), 0.0))
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] == 2
    source.assign(np.stack([a.real, a.imag], axis=-1))
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] == 0
    v, w = output(solver.vectors), output(solver.values)
    assert np.max(np.linalg.norm(a @ v - v * w, axis=0)) < 1e-10


@wp.kernel
def decrease(running: wp.array[wp.int32]):
    running[0] -= 1


@pytest.mark.parametrize("case", ["direct", "shift"])
def test_actual_projection_in_nested_capture(case, tmp_path, solver_type):
    with np.load(Path(__file__).parent / "fixtures/general_projections.npz") as data:
        a = data[case]
    source, solver = workspace(a, solver_type)
    running = wp.zeros(1, dtype=wp.int32)

    def body():
        wp.copy(solver.h, source)
        solver.run()
        wp.launch(decrease, 1, [running])

    wp.load_module(module=decrease.module, device=source.device)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        running.fill_(2)
        wp.capture_while(running, body)
    for scale in (1.0, 10.0):
        matrix = a * scale
        source.assign(np.stack([matrix.real, matrix.imag], axis=-1))
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == 0
        assert running.numpy()[0] == 0
        v, w = output(solver.vectors), output(solver.values)
        norm = np.linalg.norm(matrix)
        assert np.max(np.linalg.norm(matrix @ v - v * w, axis=0)) / norm < 1e-12
        cost = abs(w[:, None] - la.eigvals(matrix))
        rows, cols = linear_sum_assignment(cost)
        assert cost[rows, cols].max() / norm < 1e-11
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
    dot = tmp_path / "nested.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    audit = graph_summary(dot)
    expected = 2 if solver_type is TiledSchur else 1
    assert audit["node_types_including_conditional_body"].get("CONDITIONAL", 0) == expected
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


@pytest.mark.parametrize("kind", ["identity", "zero", "repeated", "nonnormal"])
def test_degenerate_projections(kind, solver_type):
    n = 32
    if kind == "zero":
        a = np.zeros((n, n), dtype=complex)
    elif kind == "identity":
        a = np.eye(n, dtype=complex)
    else:
        values = np.repeat(np.arange(1.0, n // 4 + 1), 4)
        a = np.diag(values).astype(complex)
        if kind == "nonnormal":
            a += np.diag(np.full(n - 1, 0.05 + 0.1j), 1)
        rng = np.random.default_rng(321)
        q = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
        a = q @ a @ q.conj().T
    source, solver = workspace(a, solver_type)
    wp.copy(solver.h, source)
    solver.run()
    assert solver.status.numpy()[0] == 0
    v, w = output(solver.vectors), output(solver.values)
    # Repeated defective eigenvalues are ill-conditioned; check backward error,
    # not an unjustified forward-eigenvalue or eigenbasis-orthogonality bound.
    assert np.max(np.linalg.norm(a @ v - v * w, axis=0)) / max(1, np.linalg.norm(a)) < 1e-12
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)


def test_fixed_general_capture_without_conditional_support(monkeypatch, tmp_path):
    from benchmarks.upstream.adapters import complex_matrix
    from warpack import GeneralEigensolver

    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: False)
    values = np.arange(1.0, 25)
    solver = GeneralEigensolver(complex_matrix(np.diag(values)), 4, ncv=24, tol=1e-10)
    graph = solver.capture(2, adaptive=False)
    wp.capture_launch(graph)
    assert solver.converged.numpy()[0] == 4
    np.testing.assert_allclose(np.sort(output(solver.eigenvalues).real), values[-4:], atol=1e-11)
    dot = tmp_path / "fixed.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert graph_summary(dot)["node_types_including_conditional_body"].get("CONDITIONAL", 0) == 0
