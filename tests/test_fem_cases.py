"""Elastic ROM fixtures: constraints, rigid spaces and complete physical clusters."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.applications.fem_cases import beam_mesh, elastic_matrices, make_fem_case
from benchmarks.applications.runner import accuracy, warp_output, warp_setup

NAMES = [
    "clamped_lumped",
    "clamped_consistent",
    "free",
    "disconnected",
    "heterogeneous",
    "repeated_bending",
]


@pytest.mark.parametrize("name", NAMES)
def test_fem_reference_constraints_nullspace_and_cluster(name):
    c = make_fem_case(name)
    p = c.problem
    assert abs(p.a - p.a.T).max() < 1e-12
    assert abs(p.b - p.b.T).max() < 1e-12
    la.cholesky(p.b.toarray())
    assert np.count_nonzero(abs(c.spectrum) < 1e-8) == p.nullity
    assert c.spectrum[p.nullity] > 1e-4
    assert c.spectrum[p.k] - c.spectrum[p.k - 1] > 1e-8
    np.testing.assert_allclose(p.a @ p.null_basis, 0, atol=1e-12)
    np.testing.assert_allclose(p.null_basis.T @ (p.b @ p.null_basis), np.eye(p.nullity), atol=1e-12)
    assert accuracy(p, c.spectrum[: p.k], c.vectors[:, : p.k])["accepted"]
    if name == "repeated_bending":
        np.testing.assert_array_equal(c.spectrum[0::2], c.spectrum[1::2])
        # First cantilever continuum eigenvalue beta_1^4; P1/Hermite discretization error is separate.
        assert abs(c.spectrum[0] / 1.875104068711961**4 - 1) < 5e-6
    if name.startswith("clamped") or name == "heterogeneous":
        expanded = np.zeros((3 * len(c.vertices), p.k))
        expanded[c.dofs] = c.vectors[:, : p.k]
        assert np.max(abs(expanded.reshape(-1, 3, p.k)[c.vertices[:, 0] == 0])) == 0


def test_independent_tet_strain_assembly_matches_warp_rest_tangent():
    from benchmarks.elasticity_reference import assemble_consistent_mass
    from tests.test_fem import dense
    from warpack.fem import assemble_rest

    vertices, tets = beam_mesh(1)
    a, b = elastic_matrices(vertices, tets, consistent=False)
    h, m, _ = assemble_rest(vertices, tets, young=10, density=1)
    np.testing.assert_allclose(dense(h), a.toarray(), atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(dense(m), b.toarray(), atol=2e-14, rtol=2e-14)
    _, consistent = elastic_matrices(vertices, tets, consistent=True)
    np.testing.assert_allclose(
        dense(assemble_consistent_mass(vertices, tets, density=1)),
        consistent.toarray(),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(consistent.sum(axis=1), b.sum(axis=1), atol=2e-14)


@pytest.mark.parametrize("name", NAMES)
def test_elastic_modes_captured_replay(name):
    c = make_fem_case(name)
    p = c.problem
    solver, inverse = warp_setup(p, ncv=48, tol=1e-9)
    graph = solver.capture(30)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v, right = warp_output(p, solver)
            check = accuracy(p, w, v, right)
            assert check["accepted"], check
            assert check["requested_subspace_error"] < 1e-8
            assert check["residual"] < 1e-9
            assert solver.converged.numpy()[0] == p.k
            physical = v[:, np.argsort(w)[p.nullity :]]
            np.testing.assert_allclose(p.null_basis.T @ (p.b @ physical), 0, atol=1e-9)
    finally:
        solver.graph = graph = None
        inverse.close()


def test_cluster_request_is_extended_instead_of_split():
    c = make_fem_case("repeated_bending", modes=5)
    assert c.nontrivial_modes == 6 and c.problem.k == 6


@pytest.mark.parametrize("method", ["cg-bj", "cg-fsai"])
def test_elastic_modes_with_upstream_preconditioners(method):
    import inspect

    import warp.optim.linear as linear

    if method == "cg-bj" and "block_jacobi_sequential" not in inspect.getsource(
        linear.preconditioner
    ):
        pytest.skip("requires upstream Warp block-Jacobi")
    if method == "cg-fsai" and not hasattr(linear, "FSAI"):
        pytest.skip("requires the pinned upstream FSAI branch")
    c = make_fem_case("heterogeneous", side=1)
    p = c.problem
    solver, inverse = warp_setup(p, method=method, ncv=32, tol=1e-9)
    graph = solver.capture(10)
    inverse.reset_stats()
    wp.capture_launch(graph)
    check = accuracy(p, *warp_output(p, solver))
    assert check["accepted"], check
    assert solver.converged.numpy()[0] == p.k
    assert inverse.stats.numpy()[2] == 0
