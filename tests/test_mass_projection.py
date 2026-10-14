"""Mass projection uses different primal/dual projectors for non-diagonal M."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.applications.fem_cases import make_fem_case
from benchmarks.applications.fem_projection import device_rigid_rows, physical_problem
from benchmarks.applications.runner import accuracy
from benchmarks.upstream.adapters import matrix
from tests.test_solver import sparse
from warpack import (
    CGInverse,
    CuDSSInverse,
    GeneralizedKrylovSchur,
    MassOrthogonalComplementInverse,
    SparseOperator,
)


def orthonormal_rows(raw, mass):
    lower = la.cholesky(raw.T @ mass @ raw, lower=True)
    return la.solve_triangular(lower, raw.T, lower=True)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_weighted_primal_dual_projection_and_refresh_in_graph(alias, normalize):
    rng = np.random.default_rng(421)
    n, rank = 31, 4
    m = rng.normal(size=(n, n))
    m = m.T @ m + np.eye(n)
    inv = rng.normal(size=(n, n))
    inv = inv @ inv.T + np.eye(n)
    mass = SparseOperator(sparse(m))
    basis = wp.empty((rank, n), dtype=wp.float64)
    rows = orthonormal_rows(rng.normal(size=(n, rank)), m)
    wp.copy(basis, wp.array(rows, dtype=wp.float64))
    projected = MassOrthogonalComplementInverse(
        SparseOperator(sparse(inv)), mass, basis, normalize=normalize
    )
    rhs = wp.zeros((3, n), dtype=wp.float64)
    answer = rhs if alias else wp.empty_like(rhs)
    projected.apply(rhs, answer)
    with wp.ScopedCapture() as cap:
        projected.refresh()
        projected.apply(rhs, answer)
    for scale in [1.0, 2.0]:
        current_mass = scale * m
        replacement = sparse(current_mass)
        wp.copy(mass.matrix.values, replacement.values)
        rows = orthonormal_rows(rng.normal(size=(n, rank)), current_mass)
        supplied = rng.normal(size=(rank, rank)) @ rows if normalize else rows
        wp.copy(basis, wp.array(supplied, dtype=wp.float64))
        b = rng.normal(size=(3, n))
        wp.copy(rhs, wp.array(b, dtype=wp.float64))
        wp.capture_launch(cap.graph)
        p = np.eye(n) - rows.T @ rows @ current_mass
        expected = p @ inv @ p.T
        np.testing.assert_allclose(p @ p, p, atol=1e-13)
        assert np.linalg.norm(p - p.T) > 0.1  # Swapping the dual projection is observable.
        np.testing.assert_allclose(answer.numpy(), b @ expected.T, atol=2e-11, rtol=2e-12)
        np.testing.assert_allclose(rows @ current_mass @ answer.numpy().T, 0, atol=3e-11)
        assert projected.basis_status.numpy()[0] == 0
        np.testing.assert_array_equal(basis.numpy(), supplied)
    # Wrong normalization is diagnosed on device on the same captured path.
    invalid = 2 * rows
    if normalize:
        invalid[1] = invalid[0]
    wp.copy(basis, wp.array(invalid, dtype=wp.float64))
    wp.copy(rhs, wp.array(b, dtype=wp.float64))
    wp.capture_launch(cap.graph)
    assert projected.basis_status.numpy()[0] == 1
    assert np.isnan(answer.numpy()).all()


@pytest.mark.parametrize("name", ["free", "disconnected"])
@pytest.mark.parametrize("backend", ["cudss", "cg"])
def test_projected_consistent_mass_elastic_modes(name, backend):
    c = make_fem_case(name)
    p = c.problem
    a, m = SparseOperator(matrix(p.a)), SparseOperator(matrix(p.b))
    shifted = matrix(p.a - p.target * p.b)
    inverse = (
        CuDSSInverse(shifted, 1)
        if backend == "cudss"
        else CGInverse(shifted, tol=1e-13, maxiter=2000, preconditioner="diag")
    )
    projected = MassOrthogonalComplementInverse(
        inverse, m, device_rigid_rows(c, inverse.device), normalize=True
    )
    physical = physical_problem(c)
    solver = GeneralizedKrylovSchur(a, m, projected, physical.k, ncv=48, target=p.target, tol=1e-9)
    graph = solver.capture(10)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            check = accuracy(physical, w, v)
            assert check["accepted"], check
            assert projected.basis_status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == physical.k
            np.testing.assert_allclose(p.null_basis.T @ (p.b @ v), 0, atol=1e-11)
    finally:
        solver.graph = graph = None
        if hasattr(inverse, "close"):
            inverse.close()


@pytest.mark.parametrize("bad", ["shape", "rank", "dtype", "width", "tolerance", "mass"])
def test_mass_projection_layout_contracts(bad):
    inverse = SparseOperator(sparse(np.eye(8)))
    mass = SparseOperator(sparse(np.eye(8 if bad != "mass" else 7)))
    shape = (0, 8) if bad == "rank" else (2, 7) if bad == "shape" else (2, 8)
    basis = wp.zeros(shape, dtype=wp.float32 if bad == "dtype" else wp.float64)
    if bad == "width":
        inverse.width = 2
    with pytest.raises(ValueError):
        MassOrthogonalComplementInverse(
            inverse, mass, basis, basis_tolerance=0 if bad == "tolerance" else 1e-10
        )
