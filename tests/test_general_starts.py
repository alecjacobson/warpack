"""Replayable starts for nonsymmetric direct and inverse iteration."""

import re
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.general_reference import GeneralReference
from warpack import GeneralEigensolver, ShiftInvertEigensolver


def device_vector(values):
    values = np.asarray(values, dtype=complex)
    return wp.array(np.column_stack([values.real, values.imag]), dtype=wp.vec2d)


def problem(n=18):
    values = np.arange(1.0, n + 1) * (1.0 + 0.1j)
    return np.diag(values) + np.diag(np.full(n - 1, 0.3 + 0.2j), 1)


def check_pairs(a, solver, sigma=None):
    values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
    reference, basis = la.eig(a)
    order = np.argsort(-abs(reference) if sigma is None else abs(reference - sigma))[: solver.k]
    rows, cols = linear_sum_assignment(abs(values[:, None] - reference[order]))
    assert np.max(abs(values[rows] - reference[order[cols]])) < 1e-9
    assert np.max(np.linalg.norm(a @ vectors - vectors * values, axis=0)) < 1e-9
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1, atol=1e-12)
    allowed = la.orth(basis[:, order])
    assert np.linalg.norm(vectors - allowed @ (allowed.conj().T @ vectors), 2) < 1e-9
    assert la.svdvals(vectors)[-1] > 0.1
    assert solver.status.numpy()[0] == 0
    assert solver.converged.numpy()[0] == solver.k


def test_general_start_initialization_scale_and_state_replay():
    n = 24
    rng = np.random.default_rng(62)
    initial = rng.normal(size=n) + 1j * rng.normal(size=n)
    start = device_vector(initial)
    solver = GeneralEigensolver(complex_matrix(problem(n)), 3, ncv=16, initial_vector=start)
    solver.initialize()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.initialize()
    for vector in (
        initial,
        initial.conj() * 1j,
        initial * 1e-300,
        initial * 1e300,
        initial * 1e-320,
    ):
        packed = np.column_stack([vector.real, vector.imag])
        start.assign(packed)
        solver.restart_index.fill_(0)
        solver.residual_restarts.fill_(99)
        solver.residual_fallbacks.fill_(88)
        solver.status.fill_(2)
        wp.capture_launch(capture.graph)
        assert solver.restart_index.numpy()[0] == -1
        assert solver.residual_restarts.numpy()[0] == 0
        assert solver.residual_fallbacks.numpy()[0] == 0
        assert solver.status.numpy()[0] == 0
        q = output(solver.q)[: solver.keep]
        # Component division avoids NumPy complex division overflowing its
        # reciprocal for subnormal input. Match the represented input vector.
        scale = max(abs(vector.real).max(), abs(vector.imag).max())
        normalized = vector.real / scale + 1j * (vector.imag / scale)
        normalized /= np.linalg.norm(normalized)
        np.testing.assert_allclose(q[0], normalized, atol=1e-14)
        np.testing.assert_allclose(q.conj() @ q.T, np.eye(solver.keep), atol=1e-12)
        np.testing.assert_array_equal(start.numpy(), packed)


@pytest.mark.parametrize("backend", ["direct", "cudss", "gmres"])
def test_general_changed_start_invalid_recovery_and_graph(backend, tmp_path):
    a = problem()
    rng = np.random.default_rng(83)
    initial = rng.normal(size=len(a)) + 1j * rng.normal(size=len(a))
    start = device_vector(initial)
    sigma = None if backend == "direct" else 3.4 + 0.25j
    if backend == "direct":
        solver = GeneralEigensolver(
            complex_matrix(a), 3, ncv=len(a), tol=1e-10, initial_vector=start
        )
    else:
        solver = ShiftInvertEigensolver(
            complex_matrix(a),
            3,
            sigma,
            ncv=len(a),
            backend=backend,
            tol=1e-10,
            inner_tol=1e-13,
            inner_maxiter=80,
            initial_vector=start,
        )
    graph = solver.capture(3)
    try:
        for vector in (initial, 1j * initial.conj() * 1e-250, initial * 1e250):
            packed = np.column_stack([vector.real, vector.imag])
            start.assign(packed)
            wp.capture_launch(graph)
            check_pairs(a, solver, sigma)
            np.testing.assert_array_equal(start.numpy(), packed)
        # Reuse the same graph after invalid data; a valid Schur solve must not
        # erase the start error, and neither physical nor core counts may pass.
        for bad in (0.0, float("nan"), float("inf"), -float("inf")):
            invalid = np.zeros((len(a), 2))
            invalid[0, 1] = bad
            start.assign(invalid)
            wp.capture_launch(graph)
            assert solver.status.numpy()[0] == 1
            assert solver.converged.numpy()[0] == 0
            if backend != "direct":
                assert solver.core.converged.numpy()[0] == 0
            start.assign(np.column_stack([initial.real, initial.imag]))
            wp.capture_launch(graph)
            check_pairs(a, solver, sigma)
        dot = tmp_path / "general-start.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        audit = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', audit) and "DtoH" not in audit
        if backend != "cudss":
            assert "HtoD" not in audit
    finally:
        if backend != "direct":
            solver.graph = solver.core.graph = graph = None
            solver.close()


@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("kind", ["dtype", "length", "ndim", "device", "object"])
def test_general_start_contract(shifted, kind):
    op = complex_matrix(problem(10))
    if kind == "dtype":
        start = wp.ones(10, dtype=wp.float64)
    elif kind == "length":
        start = wp.ones(9, dtype=wp.vec2d)
    elif kind == "ndim":
        start = wp.ones((1, 10), dtype=wp.vec2d)
    elif kind == "device":
        start = wp.ones(10, dtype=wp.vec2d, device="cpu")
    else:
        start = True
    solver = ShiftInvertEigensolver if shifted else GeneralEigensolver
    with pytest.raises(ValueError, match="initial_vector"):
        solver(op, 2, initial_vector=start)


def test_public_start_spectra_lm_fixture_explicit_wider_profile():
    fixture = np.load(Path(__file__).parent / "fixtures/spectra_general_lm_selection.npz")
    a, initial = fixture["a"], fixture["v0"]
    solver = GeneralEigensolver(
        complex_matrix(a), 10, ncv=40, tol=1e-10, initial_vector=device_vector(initial.ravel())
    )
    graph = solver.capture(300)
    reference = GeneralReference(a, 10)
    for _ in range(2):
        wp.capture_launch(graph)
        assert reference.check(output(solver.eigenvalues), output(solver.eigenvectors).T)[
            "accepted"
        ]
        assert solver.converged.numpy()[0] == 10
        assert solver.status.numpy()[0] == 0
