"""Original generalized equations, finite selection and replay/inverse contracts."""

import re
import weakref

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.upstream.adapters import complex_matrix, output
from warpack import ComplexShiftInverse
from warpack.pencil import GeneralizedShiftInvertEigensolver


def packed(a):
    return np.stack([np.asarray(a).real, np.asarray(a).imag], axis=-1)


def pencil(kind, n=18):
    rng = np.random.default_rng(201)
    q = la.qr(rng.normal(size=(n, n)))[0]
    r = q @ (np.eye(n) + np.diag(np.full(n - 1, 0.1), 1))
    values = np.arange(n, dtype=float).astype(complex)
    values[:4] = [0, 1, 1, 2]
    if kind == "complex":
        values += 0.08j * np.arange(n)
        values[2] = values[1]
    if kind == "singular":
        b = q @ np.diag(np.r_[np.ones(n - 2), [0, 0]]) @ q.T
        a = q @ np.diag(values + 1) @ q.T
    else:
        b = np.diag(np.linspace(1, 2, n)) + np.diag(np.full(n - 1, 0.05), 1)
        b = (b + np.diag(np.full(n - 1, 0.05), -1)).astype(complex)
        if kind == "nonsymmetric-mass":
            b += np.diag(np.full(n - 1, 0.2j), 1)
        a = b @ r @ np.diag(values) @ la.inv(r)
    return a.astype(complex), b.astype(complex)


def check(a, b, solver, sigma):
    w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
    exact, basis = la.eig(a, b)
    selected = np.argsort(abs(exact - sigma))[: solver.k]
    cost = abs(w[:, None] - exact[selected])
    rows, cols = linear_sum_assignment(cost)
    assert cost[rows, cols].max() < 1e-9
    av, bv = a @ v, b @ v
    residuals = la.norm(av - bv * w, axis=0) / (
        la.norm(av, axis=0) + np.maximum(1, abs(w)) * la.norm(bv, axis=0)
    )
    assert residuals.max() < 1e-10
    np.testing.assert_allclose(solver.residuals.numpy(), residuals, atol=1e-13)
    np.testing.assert_allclose(la.norm(v, axis=0), 1, atol=1e-12)
    allowed = la.orth(basis[:, selected])
    assert la.norm(v - allowed @ (allowed.conj().T @ v)) < 1e-8
    assert la.svdvals(v)[-1] > 0.05
    assert solver.converged.numpy()[0] == solver.k
    assert solver.status.numpy()[0] == 0


@pytest.mark.parametrize("kind", ["real", "complex", "singular", "nonsymmetric-mass"])
@pytest.mark.parametrize("backend", ["cudss", "gmres"])
def test_original_pencil_finite_modes(kind, backend, tmp_path):
    a, b = pencil(kind)
    sigma = 0.3 + 0.12j
    inverse = ComplexShiftInverse(
        complex_matrix(a - sigma * b), backend=backend, inner_tol=1e-14, inner_maxiter=120
    )
    solver = GeneralizedShiftInvertEigensolver(
        complex_matrix(a), complex_matrix(b), inverse, 4, sigma=sigma, ncv=len(a), tol=1e-10
    )
    try:
        graph = solver.capture(3)
        for _ in range(2):
            wp.capture_launch(graph)
            check(a, b, solver, sigma)
        assert solver.core._checkpoint.valid.numpy()[0] == 1
        dot = tmp_path / "pencil.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        text = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', text) and "DtoH" not in text
        if backend == "gmres":
            assert "HtoD" not in text
    finally:
        solver.close()
        graph = None
        inverse.close()


def test_current_values_starts_scaling_and_invalid_recovery():
    a, b = pencil("complex")
    sigma = 0.3 + 0.12j
    op, mass, shifted = (complex_matrix(x) for x in (a, b, a - sigma * b))
    inverse = ComplexShiftInverse(shifted, backend="gmres", inner_tol=1e-14, inner_maxiter=120)
    start = wp.array(packed(np.ones(len(a), dtype=complex)), dtype=wp.vec2d)
    solver = GeneralizedShiftInvertEigensolver(
        op, mass, inverse, 4, sigma=sigma, ncv=len(a), tol=1e-10, initial_vector=start
    )
    try:
        graph = solver.capture(3)
        for shift, scale in ((0.0, 1.0), (0.2, 3.0), (0.0, 1.0)):
            changed = a + shift * b
            op.values.assign(packed((scale * changed).ravel()))
            # Mass is tridiagonal; retain its original compact CSR structure.
            mass.values.assign(packed(scale * b[b != 0]))
            shifted.values.assign(packed((scale * (changed - sigma * b)).ravel()))
            wp.capture_launch(graph)
            check(scale * changed, scale * b, solver, sigma)
            wp.copy(start, solver.eigenvectors[0])
        original_start = start.numpy().copy()
        for bad in (0.0, float("nan"), float("inf")):
            invalid = np.zeros((len(a), 2))
            invalid[0, 0] = bad
            start.assign(invalid)
            wp.capture_launch(graph)
            assert solver.status.numpy()[0] == 1
            assert solver.converged.numpy()[0] == solver.core.converged.numpy()[0] == 0
            np.testing.assert_array_equal(start.numpy(), invalid)
            start.assign(original_start)
            wp.capture_launch(graph)
            check(a, b, solver, sigma)
    finally:
        solver.close()
        graph = None
        inverse.close()


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_physical_checkpoint_gate_ignores_small_transformed_error(scale):
    b = np.diag(np.arange(1.0, 9.0)) * scale
    a = b @ np.diag(np.arange(8.0))
    sigma = -0.5
    inverse = ComplexShiftInverse(complex_matrix(a - sigma * b), backend="gmres")
    solver = GeneralizedShiftInvertEigensolver(
        complex_matrix(a), complex_matrix(b), inverse, 2, sigma=sigma, ncv=6, tol=1e-10
    )
    try:
        core = solver.core
        core._enable_checkpoint(True)
        core.order.assign(np.arange(6, dtype=np.int32))
        core.eigenvectors.assign(packed(np.eye(8)[:2]))
        for mu, good in (([20, 30], False), ([2, 2 / 3], True), ([0, 0], False)):
            core.ritz.assign(packed(np.r_[mu, np.ones(4)]))
            core.norms.zero_()
            previous = core._checkpoint.updates.numpy()[0]
            core._checkpoint.consider(core)
            assert core._checkpoint.updates.numpy()[0] == previous + int(good)
        core.finalize()
        solver.finalize()
        np.testing.assert_allclose(output(solver.eigenvalues), [0, 1], atol=1e-12)
        assert solver.converged.numpy()[0] == 2
        # A cached accepted set must not bypass a changed original operator.
        solver.op.values.assign(packed(a[a != 0] * 2 + scale))
        solver.finalize()
        assert solver.converged.numpy()[0] == 1  # unchanged exact null mode only
    finally:
        solver.close()
        inverse.close()


def test_borrowed_inverse_and_reference_lifetime():
    a, b = pencil("real")
    inverse = ComplexShiftInverse(complex_matrix(a + b), backend="gmres")
    solver = GeneralizedShiftInvertEigensolver(
        complex_matrix(a), complex_matrix(b), inverse, 4, sigma=-1, ncv=len(a)
    )
    solver.capture(2)
    ref = weakref.ref(solver)
    solver.close()
    del solver
    assert ref() is None
    # A different owner can still use the borrowed inverse after close.
    rhs = wp.ones((1, len(a)), dtype=wp.vec2d)
    result = wp.empty_like(rhs)
    inverse.apply_complex(rhs, result)
    np.testing.assert_allclose((a + b) @ output(result)[0], output(rhs)[0], atol=1e-9)
    inverse.close()


@pytest.mark.parametrize("bad", ["size", "device", "protocol", "sigma", "tol", "start"])
def test_pencil_contract_rejection(bad):
    op = complex_matrix(np.eye(8))
    mass, inverse, kwargs = op, op, {}
    if bad == "size":
        mass = complex_matrix(np.eye(7))
    elif bad == "device":
        with wp.ScopedDevice("cpu"):
            mass = complex_matrix(np.eye(8))
    elif bad == "protocol":
        inverse = object()
    elif bad == "sigma":
        kwargs["sigma"] = complex(0, np.nan)
    elif bad == "tol":
        kwargs["tol"] = 0
    else:
        kwargs["initial_vector"] = wp.ones(8, dtype=wp.float64)
    with pytest.raises((ValueError, TypeError)):
        GeneralizedShiftInvertEigensolver(op, mass, inverse, 2, ncv=6, **kwargs)
