"""Independent assembly, signed selection, and original buckling-space checks."""

import numpy as np
import pytest
import scipy.linalg as la

from benchmarks.applications.fem_cases import beam_mesh
from benchmarks.applications.prestress_buckling import geometric_stiffness, make_buckling_case
from benchmarks.applications.runner import accuracy


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("side", [1, 2])
def test_signed_tet_pencil_and_complete_reference(side, sign):
    case = make_buckling_case(side, 4, sign)
    p = case.problem
    k, g = p.a.toarray(), p.b.toarray()
    np.testing.assert_allclose(k, k.T, atol=1e-14)
    np.testing.assert_allclose(g, g.T, atol=1e-14)
    assert la.eigvalsh(k)[0] > 0
    geometric = la.eigvalsh(g)
    assert geometric[0] < 0 < geometric[-1]
    assert np.count_nonzero(k - np.diag(k.diagonal())) > 0
    assert np.count_nonzero(g - np.diag(g.diagonal())) > 0
    assert la.eigvalsh(k - p.target * g)[0] > 0
    assert case.shifted_spd and case.selection_margin > 0
    # Independent nonsymmetric generalized QZ, without assuming G is SPD.
    values = la.eigvals(k, g)
    assert abs(values.imag).max() < 1e-9
    values = values.real
    selected = np.argsort(-abs(values / (values - p.target)))[: p.k]
    np.testing.assert_allclose(np.sort(p.expected), np.sort(values[selected]), rtol=1e-10)
    nearest = values[np.argsort(abs(values - p.target))[: p.k]]
    assert not np.allclose(np.sort(p.expected), np.sort(nearest))
    assert np.all(sign * p.expected > 0)
    assert accuracy(p, p.expected, p.requested_basis)["accepted"]
    # Rotating the complete requested basis cannot change its space.
    q, _ = la.qr(np.random.default_rng(812).normal(size=(p.k, p.k)))
    old = p.requested_basis.copy()
    p.requested_basis = old @ q
    assert accuracy(p, p.expected, old)["accepted"]


def test_geometric_energy_from_affine_tet_deformations():
    vertices, tets = beam_mesh(2)
    stress = np.array([[1.0, 0.2, 0.1], [0.2, -0.6, 0.05], [0.1, 0.05, 0.3]])
    g = geometric_stiffness(vertices, tets, stress)
    displacement = np.random.default_rng(91).normal(size=vertices.shape)
    total = 0.0
    for tet in tets:
        x, u = vertices[tet], displacement[tet]
        # Fit u(x)=c+F*x directly, independent of assembly's shape gradients.
        coefficients = la.solve(np.column_stack([np.ones(4), x]), u)
        gradient = coefficients[1:].T
        volume = abs(la.det((x[1:] - x[0]).T)) / 6
        total += 0.5 * volume * np.trace(gradient @ stress @ gradient.T)
    flat = displacement.ravel()
    assert 0.5 * flat @ (g @ flat) == pytest.approx(total, rel=1e-12, abs=1e-12)
    for axis in np.eye(3):
        translation = np.broadcast_to(axis, vertices.shape).ravel()
        np.testing.assert_allclose(g @ translation, 0.0, atol=1e-14)


@pytest.mark.parametrize("shift", [0.0, -1.0, np.inf, 100.0])
def test_invalid_or_past_first_factor_shift_rejected(shift):
    with pytest.raises(ValueError):
        make_buckling_case(1, 4, 1, shift)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("method", ["cudss", "cg"])
def test_captured_signed_tet_buckling(sign, method, tmp_path):
    import warp as wp

    from benchmarks.audit_profile import graph_summary
    from benchmarks.upstream.adapters import matrix
    from warpack import BucklingEigensolver, CGInverse, CuDSSInverse, SparseOperator

    case = make_buckling_case(2, 4, sign)
    p = case.problem
    shifted = matrix(p.a - p.target * p.b)
    inverse = (
        CuDSSInverse(shifted, 1, mtype="spd", refinement_steps=1)
        if method == "cudss"
        else CGInverse(shifted, tol=1e-14, atol=0.0, maxiter=1000, preconditioner="diag")
    )
    solver = BucklingEigensolver(
        SparseOperator(matrix(p.a)),
        SparseOperator(matrix(p.b)),
        inverse,
        p.k,
        p.target,
        ncv=24,
        tol=1e-10,
        method="krylov",
    )
    graph = solver.capture(5)
    try:
        for _ in range(3):
            if method == "cg":
                inverse.reset_stats()
            wp.capture_launch(graph)
            q = accuracy(p, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
            assert q["accepted"], q
            assert solver.converged.numpy()[0] == p.k
            assert solver.core.converged.numpy()[0] == p.k
            assert solver.core.status.numpy()[0] == 0
            if method == "cg":
                assert inverse.stats.numpy()[2] == 0
        path = tmp_path / f"buckling-{sign}-{method}.dot"
        wp.capture_debug_dot_print(graph, str(path))
        audit = graph_summary(path)
        assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
        if method == "cg":
            assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
    finally:
        solver.core.graph = solver.graph = graph = None
        if method == "cudss":
            inverse.close()


def test_native_exception_retains_completed_trial_and_rejects_comparison(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from benchmarks.acceptance import assess_comparison, require_accepted
    from benchmarks.applications import buckling_compare as comparison

    p = make_buckling_case(1, 4, -1).problem
    calls, closed = [], []

    def native():
        calls.append(1)
        if len(calls) == 3:
            raise AttributeError("residual orthogonalization returned None")
        return -1 / p.expected, p.requested_basis, None

    monkeypatch.setattr(
        comparison, "cupy_setup", lambda *a, **k: (native, lambda: closed.append(1), "test")
    )
    monkeypatch.setattr(comparison, "timed", lambda function: (0.1, function()))
    monkeypatch.setattr(comparison, "other_gpu_processes", lambda: [])
    args = SimpleNamespace(ncv=24, cupy_tol=1e-10, cupy_steps=400, repeats=3)
    row = comparison.gpu_run(
        p, "cupy-reciprocal", args, tmp_path / "report.json", "negative", -1, set()
    )
    assert row["outcome"] == "error" and not row["accepted"]
    assert len(row["validation"]) == len(row["samples_seconds"]) == 1
    assert row["validation"][0]["accepted"]
    assert closed == [1]
    report = {}
    assert not assess_comparison(report, [row])
    with pytest.raises(SystemExit):
        require_accepted(report)
