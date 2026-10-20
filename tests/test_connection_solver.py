"""Complex connection pencils: weighted recovery and captured replay."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.applications.connection import make_connection_case
from benchmarks.applications.connection_backends import warp_setup
from benchmarks.applications.connection_warp import ConnectionEigensolver
from benchmarks.applications.runner import accuracy
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import complex_matrix, output
from warpack import ShiftInvertEigensolver
from warpack.shift import ComplexShiftInverse


@pytest.mark.parametrize("mass_kind", ["identity", "lumped", "consistent"])
@pytest.mark.parametrize("copies", [1, 2])
def test_weighted_complex_modes_changed_start(mass_kind, copies, tmp_path):
    p = make_connection_case(side=7, modes=5, mass_kind=mass_kind, copies=copies).problem
    m = sp.eye(p.n, dtype=complex, format="csr") if p.b is None else p.b
    inverse = ComplexShiftInverse(complex_matrix(p.a - p.target * m))
    rng = np.random.default_rng(219)
    start = wp.array(rng.normal(size=2 * p.n), dtype=wp.float64)
    solver = ConnectionEigensolver(
        complex_matrix(p.a, True),
        complex_matrix(m, True),
        inverse,
        p.k,
        ncv=24,
        initial_vector=start,
        tol=1e-10,
    )
    graph = solver.capture(15)
    try:
        for _ in range(2):
            start.assign(rng.normal(size=2 * p.n))
            wp.capture_launch(graph)
            result = accuracy(p, solver.eigenvalues.numpy(), output(solver.eigenvectors).T)
            assert result["accepted"], result
            assert solver.count.numpy()[0] == p.k
            assert solver.converged.numpy()[0] == p.k
            assert solver.real.converged.numpy()[0] == 2 * p.k
            assert solver.real.status.numpy()[0] == 0
        dot = tmp_path / "connection.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
        assert not any("DtoH" in kind for kind in audit["memcpy_kinds"])
    finally:
        solver.real.graph = graph = None
        inverse.close()


def test_missing_complex_directions_cannot_pass_finalizer():
    p = make_connection_case(side=4, modes=3).problem
    inverse = ComplexShiftInverse(complex_matrix(p.a - p.target * p.b))
    solver = ConnectionEigensolver(
        complex_matrix(p.a, True), complex_matrix(p.b, True), inverse, p.k, ncv=12
    )
    try:
        solver.real.eigenvectors.zero_()
        solver.real.bx.zero_()
        solver.real.eigenvalues.zero_()
        solver.finalize()
        assert solver.count.numpy()[0] == 0
        assert solver.converged.numpy()[0] == 0
        assert np.all(np.isinf(solver.residuals.numpy()))
    finally:
        inverse.close()


@pytest.mark.parametrize("copies", [1, 2])
def test_standard_public_complex_api(copies):
    p = make_connection_case(side=7, modes=5, mass_kind="identity", copies=copies).problem
    solver = ShiftInvertEigensolver(
        complex_matrix(p.a), p.k, p.target, ncv=24, hermitian=True, tol=1e-10
    )
    try:
        graph = solver.capture(15)
        wp.capture_launch(graph)
        result = accuracy(p, output(solver.eigenvalues).real, output(solver.eigenvectors).T)
        assert result["accepted"], result
        assert solver.converged.numpy()[0] == p.k
        assert solver.core.count.numpy()[0] == p.k
    finally:
        graph = None
        solver.close()


@pytest.mark.parametrize("mass_kind", ["identity", "lumped", "consistent"])
def test_pure_warp_cg_connection_graph(mass_kind, tmp_path):
    p = make_connection_case(side=6, modes=5, mass_kind=mass_kind, copies=2).problem
    solver, inverse = warp_setup(p, 24, "cg", cg_tol=1e-12)
    graph = solver.capture(15)
    for _ in range(2):
        inverse.reset_stats()
        wp.capture_launch(graph)
        result = accuracy(p, solver.eigenvalues.numpy(), output(solver.eigenvectors).T)
        assert result["accepted"], result
        assert solver.converged.numpy()[0] == p.k
        assert inverse.stats.numpy()[2] == 0
        assert inverse.fresh_stats.numpy()[2] == 0
    dot = tmp_path / "connection-cg.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


def test_short_budget_is_not_a_successful_requested_spectrum():
    p = make_connection_case(side=12, modes=8).problem
    solver, inverse = warp_setup(p, 10)
    try:
        graph = solver.capture(1)
        wp.capture_launch(graph)
        result = accuracy(p, solver.eigenvalues.numpy(), output(solver.eigenvectors).T)
        assert not result["accepted"]
        assert solver.converged.numpy()[0] < p.k
    finally:
        solver.real.graph = graph = None
        inverse.close()


def test_nonfinite_start_cannot_certify_then_same_graph_recovers():
    p = make_connection_case(side=7, modes=5).problem
    solver, inverse = warp_setup(p, 24)
    graph = solver.capture(15)
    initial = solver.real.initial_vector
    good = initial.numpy()
    try:
        initial.fill_(float("nan"))
        wp.capture_launch(graph)
        assert solver.converged.numpy()[0] == 0
        initial.assign(good)
        wp.capture_launch(graph)
        assert solver.converged.numpy()[0] == p.k
        result = accuracy(p, solver.eigenvalues.numpy(), output(solver.eigenvectors).T)
        assert result["accepted"], result
    finally:
        solver.real.graph = graph = None
        inverse.close()
