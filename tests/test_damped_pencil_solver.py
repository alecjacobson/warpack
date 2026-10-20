"""Sparse consistent-mass right modes with direct and iterative device inverses."""

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.damped import quality
from benchmarks.applications.damped_backends import warp_setup
from benchmarks.applications.damped_pencil import consistent_chain
from benchmarks.experimental.pencil_polish import PolishedPencil
from benchmarks.upstream.adapters import complex_matrix, output
from warpack import ComplexShiftInverse
from warpack.pencil import GeneralizedShiftInvertEigensolver


@pytest.mark.parametrize("kind", ["nonproportional", "gyroscopic"])
@pytest.mark.parametrize("backend", ["cudss", "gmres"])
def test_consistent_damped_pencil_replay(kind, backend):
    case = consistent_chain(nodes=12 if backend == "gmres" else 24, k=6, kind=kind)
    shifted = complex_matrix(case.a - case.target * case.b)
    inverse = ComplexShiftInverse(
        shifted,
        backend=backend,
        inner_tol=1e-14,
        inner_maxiter=1000,
        inner_restart=96 if backend == "gmres" else None,
    )
    solver = GeneralizedShiftInvertEigensolver(
        complex_matrix(case.a),
        complex_matrix(case.b),
        inverse,
        case.k,
        sigma=case.target,
        ncv=32,
        tol=1e-10,
    )
    try:
        graph = solver.capture(8)
        for _ in range(2):
            wp.capture_launch(graph)
            check = quality(case, output(solver.eigenvalues), output(solver.eigenvectors).T)
            assert check["accepted"], check
            assert solver.converged.numpy()[0] == case.k
            assert solver.status.numpy()[0] == 0
            assert np.max(solver.residuals.numpy()) <= solver.tol
    finally:
        solver.close()
        graph = None
        inverse.close()


def test_pencil_quotient_keeps_zero_and_repeated_modes():
    n = 12
    b = np.diag(np.linspace(1, 2, n)) + np.diag(np.full(n - 1, 0.1), 1)
    b += np.diag(np.full(n - 1, 0.1), -1)
    exact = np.r_[0.0, 1, 1, 1, np.arange(4.0, n)]
    a = b @ np.diag(exact)
    sigma = -0.2 + 0.1j
    inverse = ComplexShiftInverse(
        complex_matrix(a - sigma * b), backend="gmres", inner_restart=24, inner_tol=1e-14
    )
    solver = PolishedPencil(
        complex_matrix(a),
        complex_matrix(b),
        inverse,
        4,
        sigma=sigma,
        ncv=n,
        tol=1e-10,
        polish_steps=1,
        rayleigh=True,
    )
    try:
        graph = solver.capture(3)
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
            np.testing.assert_allclose(np.sort(w.real), exact[:4], atol=1e-11)
            assert np.max(np.linalg.norm(a @ v - (b @ v) * w, axis=0)) < 1e-10
            assert np.linalg.svd(v, compute_uv=False)[-1] > 0.1
            assert solver.converged.numpy()[0] == 4 and solver.status.numpy()[0] == 0
    finally:
        solver.close()
        graph = None
        inverse.close()


def test_singular_pencil_cannot_report_infinite_modes_as_finite_convergence():
    n, k = 12, 4
    a = np.diag(np.arange(1.0, n + 1))
    b = np.diag(np.r_[np.ones(3), np.zeros(n - 3)])
    sigma = -0.5 + 0.2j
    inverse = ComplexShiftInverse(complex_matrix(a - sigma * b), backend="gmres", inner_restart=24)
    solver = GeneralizedShiftInvertEigensolver(
        complex_matrix(a), complex_matrix(b), inverse, k, sigma=sigma, ncv=n, tol=1e-10
    )
    try:
        graph = solver.capture(3)
        wp.capture_launch(graph)
        # The regular pencil has only three finite eigenvalues. A fourth
        # zero transformed value cannot satisfy a finite original equation.
        assert solver.converged.numpy()[0] < k
        assert not np.all(solver.residuals.numpy() <= solver.tol)
    finally:
        solver.close()
        graph = None
        inverse.close()


@pytest.mark.parametrize("polish", [1, 2])
@pytest.mark.parametrize("rayleigh", [False, True])
def test_declared_pencil_polish_preserves_physical_modes(polish, rayleigh):
    case = consistent_chain(nodes=24, k=6, kind="gyroscopic")
    solver = warp_setup(case, 32, tol=1e-10, polish=polish, pencil_rayleigh=rayleigh)
    try:
        graph = solver.capture(3)
        for _ in range(2):
            wp.capture_launch(graph)
            assert quality(case, output(solver.eigenvalues), output(solver.eigenvectors).T)[
                "accepted"
            ]
            assert solver.converged.numpy()[0] == case.k
            assert solver.status.numpy()[0] == 0
            if rayleigh:
                v = output(solver.eigenvectors).T
                av, bv = case.a @ v, case.b @ v
                expected = np.sum(bv.conj() * av, axis=0) / np.sum(abs(bv) ** 2, axis=0)
                np.testing.assert_allclose(output(solver.eigenvalues), expected, atol=1e-12)
    finally:
        solver.close()


@pytest.mark.parametrize("polish", [0, 1])
def test_consistent_pencil_kinematic_recovery_uses_fresh_equations(polish):
    case = consistent_chain(nodes=24, k=6, kind="gyroscopic")
    solver = warp_setup(case, 32, tol=1e-10, polish=polish, kinematic=True)
    try:
        graph = solver.capture(3)
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
            assert quality(case, w, v)["accepted"]
            assert solver.converged.numpy()[0] == case.k and solver.status.numpy()[0] == 0
            av, bv = case.a @ v, case.b @ v
            errors = np.linalg.norm(av - bv * w, axis=0) / (
                np.linalg.norm(av, axis=0) + np.maximum(1, abs(w)) * np.linalg.norm(bv, axis=0)
            )
            np.testing.assert_allclose(solver.residuals.numpy(), errors, atol=1e-13)
    finally:
        solver.close()
