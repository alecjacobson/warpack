"""Original-equation correction: counters, scaling, aliases and device replay."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.applications.connection import make_connection_case
from benchmarks.applications.connection_warp import ConnectionEigensolver
from benchmarks.applications.runner import accuracy
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import complex_matrix, matrix, output
from warpack import CGInverse


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_refinement_steps(steps):
    with pytest.raises((ValueError, TypeError)):
        CGInverse(object(), refinement_steps=steps)


@pytest.mark.parametrize("tol", [0.0, 1.0, -0.01, float("nan"), float("inf")])
def test_invalid_refinement_tolerance(tol):
    with pytest.raises(ValueError, match="refinement_tol"):
        CGInverse(object(), refinement_tol=tol)


@pytest.mark.parametrize("steps", [0, 1, 2])
@pytest.mark.parametrize("inplace", [False, True])
def test_changed_rhs_scaling_and_original_audit(steps, inplace, tmp_path):
    n = 129
    a = sp.diags([-np.ones(n - 1), np.full(n, 2.0), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    inverse = CGInverse(
        matrix(a),
        tol=1e-10,
        refinement_steps=steps,
        verify_residuals=True,
        atol=0.0,
        preconditioner="diag",
    )
    rhs = wp.empty((1, n), dtype=wp.float64)
    out = rhs if inplace else wp.empty_like(rhs)

    def body():
        inverse.reset_stats()
        inverse.apply(rhs, out)

    rhs.fill_(1.0)
    body()
    wp.synchronize()
    with wp.ScopedCapture() as captured:
        body()
    rng = np.random.default_rng(166)
    for scale in (1e-6, 1.0, 1e6):
        b = scale * rng.normal(size=(1, n))
        rhs.assign(b)
        wp.capture_launch(captured.graph)
        solution = out.numpy()
        np.testing.assert_allclose(solution[0], sla.spsolve(a, b[0]), rtol=1e-8, atol=1e-9 * scale)
        residual = np.linalg.norm(a @ solution[0] - b[0])
        relative = residual / np.linalg.norm(b)
        measured = inverse.residual_maxima.numpy()
        np.testing.assert_allclose(measured[0], residual, rtol=0.05, atol=1e-12 * scale)
        np.testing.assert_allclose(measured[1], relative, rtol=0.05, atol=1e-12)
        assert inverse.fresh_stats.numpy()[2] == 0
        assert relative < 1e-10
        base = inverse.initial_stats.numpy() if steps else inverse.stats.numpy()
        correction = inverse.refinement_stats.numpy() if steps else np.zeros(4, dtype=int)
        stats = inverse.stats.numpy()
        assert stats[0] == base[0] + correction[0]
        assert stats[1] == stats[0] and stats[2] == 0 and stats[3] == 1
        assert base[3] == 1 and correction[3] == steps
        np.testing.assert_array_equal(inverse.fresh_stats.numpy(), stats)
        if not inplace:
            np.testing.assert_array_equal(rhs.numpy(), b)
    dot = tmp_path / "corrected-cg.dot"
    wp.capture_debug_dot_print(captured.graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


def test_failure_then_recovery_on_same_graph():
    n = 32
    inverse = CGInverse(
        matrix(sp.eye(n, format="csr")),
        tol=1e-11,
        atol=0.0,
        refinement_steps=1,
        verify_residuals=True,
    )
    rhs = wp.ones((1, n), dtype=wp.float64)
    out = wp.empty_like(rhs)

    def body():
        inverse.reset_stats()
        inverse.apply(rhs, out)

    body()
    wp.synchronize()
    with wp.ScopedCapture() as captured:
        body()
    rhs.fill_(float("nan"))
    wp.capture_launch(captured.graph)
    assert inverse.fresh_stats.numpy()[2] == 1
    assert np.isinf(inverse.residual_maxima.numpy()).all()
    rhs.fill_(2.0)
    wp.capture_launch(captured.graph)
    np.testing.assert_allclose(out.numpy(), 2.0)
    assert inverse.fresh_stats.numpy()[2] == 0
    rhs.zero_()
    wp.capture_launch(captured.graph)
    np.testing.assert_array_equal(out.numpy(), 0.0)
    np.testing.assert_array_equal(inverse.residual_maxima.numpy(), 0.0)


def test_final_audit_uses_original_rhs_tolerance_for_multiple_scales():
    n = 129
    a = sp.diags([-np.ones(n - 1), np.full(n, 2.0), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    inverse = CGInverse(
        matrix(a),
        tol=1e-8,
        atol=0.0,
        refinement_steps=1,
        refinement_tol=1e-12,
        verify_residuals=True,
    )
    rhs = np.random.default_rng(582).normal(size=(3, n)) * np.array([1e-6, 1.0, 1e6])[:, None]
    source = wp.array(rhs, dtype=wp.float64)
    result = wp.empty_like(source)
    inverse.apply(source, result)
    residual = np.linalg.norm(a @ result.numpy().T - rhs.T, axis=0) / np.linalg.norm(rhs, axis=1)
    assert np.max(residual) < 1e-8
    assert np.max(residual) > 1e-15  # Fresh roundoff greatly exceeds the correction RHS threshold.
    assert inverse.fresh_stats.numpy()[2] == 0
    assert inverse.fresh_stats.numpy()[3] == 3
    assert inverse.initial_stats.numpy()[3] == 3
    assert inverse.refinement_stats.numpy()[3] == 3


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("workspace", ["b", "x"])
def test_exposed_workspace_inplace_refinement(audit, workspace):
    n = 32
    diagonal = np.linspace(1.0, 4.0, n)
    inverse = CGInverse(
        matrix(sp.diags(diagonal, format="csr")),
        tol=1e-11,
        atol=0.0,
        preconditioner="diag",
        refinement_steps=1,
        verify_residuals=audit,
    )
    values = getattr(inverse, workspace).reshape((1, n))
    rhs = np.arange(1.0, n + 1).reshape(1, n)
    values.assign(rhs)
    inverse.apply(values, values)
    np.testing.assert_allclose(values.numpy(), rhs / diagonal, rtol=1e-12)
    assert inverse.stats.numpy()[3] == 1
    assert inverse.initial_stats.numpy()[3] == 1
    assert inverse.refinement_stats.numpy()[3] == 1
    if audit:
        assert inverse.fresh_stats.numpy()[2] == 0
    else:
        assert inverse.fresh_stats is None


def test_corrected_inverse_in_repeated_complex_pencil_with_changed_starts():
    p = make_connection_case(side=7, modes=5, copies=2).problem
    shifted = p.a - p.target * p.b
    realified = sp.bmat([[shifted.real, -shifted.imag], [shifted.imag, shifted.real]], format="csr")
    inverse = CGInverse(
        matrix(realified),
        tol=1e-11,
        atol=0.0,
        preconditioner="diag",
        refinement_steps=1,
        verify_residuals=True,
    )
    rng = np.random.default_rng(779)
    start = wp.array(rng.normal(size=2 * p.n), dtype=wp.float64)
    solver = ConnectionEigensolver(
        complex_matrix(p.a, True),
        complex_matrix(p.b, True),
        inverse,
        p.k,
        ncv=24,
        tol=1e-10,
        initial_vector=start,
    )
    graph = solver.capture(15)
    for _ in range(2):
        start.assign(rng.normal(size=2 * p.n))
        inverse.reset_stats()
        wp.capture_launch(graph)
        check = accuracy(p, solver.eigenvalues.numpy(), output(solver.eigenvectors).T)
        assert check["accepted"], check
        assert solver.converged.numpy()[0] == p.k
        assert solver.real.converged.numpy()[0] == 2 * p.k
        assert inverse.fresh_stats.numpy()[2] == 0
        assert inverse.stats.numpy()[2] == 0
        assert inverse.refinement_stats.numpy()[3] == inverse.initial_stats.numpy()[3]
