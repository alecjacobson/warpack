"""Topology is referenced independently of residual convergence and fixed k."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.hodge import (
    make_hodge_case,
    torus_harmonics,
    torus_spectrum,
    weak_component_reference,
)
from benchmarks.applications.runner import accuracy, warp_output, warp_setup

CASES = [
    ("sphere", 0, 1, 1),
    ("disk", 0, 1, 0),
    ("annulus", 1, 1, 0),
    ("torus", 2, 1, 1),
    ("disconnected", 4, 2, 2),
    ("weak_positive", 3, 2, 1),
]


@pytest.mark.parametrize("name,beta,components,closed", CASES)
def test_independent_incidence_topology(name, beta, components, closed):
    c = make_hodge_case(name)
    p = c.problem
    assert (c.boundary1 @ c.boundary2).nnz == 0
    assert c.rank1 == c.boundary1.shape[0] - components
    assert c.rank2 == c.boundary2.shape[1] - closed
    assert p.nullity == beta
    assert np.count_nonzero(abs(c.spectrum) < 1e-8) == beta
    assert c.spectrum[beta] > 1e-8
    np.testing.assert_allclose(p.a @ p.null_basis, 0, atol=1e-12)
    np.testing.assert_allclose(p.null_basis.T @ (p.b @ p.null_basis), np.eye(beta), atol=1e-12)


@pytest.mark.parametrize("side", [3, 4, 7])
def test_torus_fourier_and_coordinate_reference(side):
    c = make_hodge_case("torus", side)
    np.testing.assert_allclose(c.spectrum, torus_spectrum(side), atol=1e-12, rtol=0)
    h = torus_harmonics(c.boundary1, side)
    np.testing.assert_array_equal(c.boundary1 @ h, 0)
    np.testing.assert_array_equal(c.boundary2.T @ h, 0)


@pytest.mark.parametrize("name", [x[0] for x in CASES])
def test_all_null_modes_and_positive_guard_in_graph(name):
    c = make_hodge_case(name)
    p = c.problem
    solver, inverse = warp_setup(p)
    graph = solver.capture(100)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v, right = warp_output(p, solver)
            check = accuracy(p, w, v, right)
            assert check["accepted"], check
            assert check["recovered_nullity"] == p.nullity
            assert check["positive_guard"] > 1e-8
            assert check["nullspace_subspace_error"] < 1e-10
            assert abs(p.a @ v - (p.b @ v) * w).max() < 1e-10
            allowed = c.vectors[:, c.spectrum <= p.expected[-1] + 1e-9]
            np.testing.assert_allclose(
                np.linalg.svd(allowed.T @ (p.b @ v), compute_uv=False),
                np.ones(p.k),
                atol=1e-9,
                rtol=0,
            )
            assert solver.converged.numpy()[0] == p.k
    finally:
        solver.graph = graph = None
        inverse.close()


def test_fixed_k_with_only_zeros_is_not_a_complete_count():
    c = make_hodge_case("torus")
    p = replace(c.problem, k=2, expected=c.spectrum[:2])
    check = accuracy(p, c.spectrum[:2], c.vectors[:, :2])
    assert check["residual"] < 1e-12 and check["spectrum_error"] < 1e-12
    assert check["recovered_nullity"] == 2
    assert check["positive_guard"] is None
    assert not check["accepted"]  # Benchmark counting contract, not an eigensolve failure.


def test_weak_positive_below_cutoff_is_unresolved_despite_exact_eigenpairs():
    c = make_hodge_case("weak_positive", weak_scale=1e-10)
    p = c.problem
    check = accuracy(p, c.spectrum[: p.k], c.vectors[:, : p.k])
    assert check["residual"] < 1e-12 and check["spectrum_error"] < 1e-12
    assert c.spectrum[p.nullity] > 0
    assert check["recovered_nullity"] > p.nullity
    assert not check["accepted"]


def test_large_disconnected_analytic_reference():
    c = make_hodge_case("large_disconnected", side=41)
    p = c.problem
    assert p.n == 10086 and p.nullity == 4 and p.k == 6
    assert (c.boundary1 @ c.boundary2).nnz == 0
    np.testing.assert_allclose(p.a @ p.null_basis, 0, atol=1e-14)
    np.testing.assert_allclose(p.null_basis.T @ p.null_basis, np.eye(4), atol=1e-12)
    assert c.spectrum[4] > 1e-8


@pytest.mark.parametrize("api_class", ["F", "I"])
def test_explicit_reference_basis_budget_for_regular_and_svd(api_class):
    from benchmarks.applications.problems import make_problem
    from benchmarks.applications.runner import scipy_setup

    p = make_problem(api_class, 80)
    assert accuracy(p, *scipy_setup(p, ncv=32)())["accepted"]


@pytest.mark.parametrize("scale", [1e-4, 1e-10, 1e-12])
def test_extreme_weak_component_retains_finite_independent_modes(scale):
    c = make_hodge_case("weak_positive", weak_scale=scale)
    p = c.problem
    expected, space = weak_component_reference(c, scale)
    solver, inverse = warp_setup(p)
    graph = solver.capture(100)
    try:
        counts = []
        for _ in range(2):
            wp.capture_launch(graph)
            w, v, _ = warp_output(p, solver)
            assert solver.status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == p.k
            counts.append(int(solver.orthogonalization_recoveries.numpy()[0]))
            assert np.isfinite(w).all() and np.isfinite(v).all()
            np.testing.assert_allclose(v.T @ (p.b @ v), np.eye(p.k), atol=1e-10)
            assert np.linalg.norm(p.a @ v - (p.b @ v) * w, axis=0).max() < 1e-10
            np.testing.assert_allclose(np.sort(w)[: p.nullity], 0, atol=1e-14)
            np.testing.assert_allclose(
                np.sort(w)[p.nullity :] / scale,
                expected[p.nullity :] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
            remainder = v - space @ (space.T @ (p.b @ v))
            assert np.linalg.norm(remainder) < 1e-9
        assert counts[0] == counts[1]  # replay resets the counter
        if scale <= 1e-10:
            assert counts[0] > 0
    finally:
        solver.graph = graph = None
        inverse.close()


@pytest.mark.parametrize("method", ["krylov", "cg", "cg-fsai"])
def test_mass_krylov_hodge_preserves_complete_harmonic_space(method):
    import warp.optim.linear as linear

    if method == "cg-fsai" and not hasattr(linear, "FSAI"):
        pytest.skip("Requires pinned upstream Warp FSAI")
    problem = make_hodge_case("large_disconnected", side=7).problem
    solver, inverse = warp_setup(problem, method, ncv=32, force_krylov=True)
    graph = solver.capture(2)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            result = accuracy(problem, *warp_output(problem, solver))
            assert result["accepted"], result
            assert result["recovered_nullity"] == 4
            assert result["positive_guard"] > 1e-8
            assert result["nullspace_subspace_error"] < 1e-9
            assert solver.converged.numpy()[0] == problem.k
            assert solver.status.numpy()[0] == 0
    finally:
        solver.graph = graph = None
        getattr(inverse, "close", lambda: None)()
