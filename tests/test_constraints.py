import numpy as np
import pytest
import warp as wp

from tests.test_solver import sparse
from warpack import CGInverse, KrylovSchur, OrthogonalComplementOperator, SparseOperator
from warpack.fem import assemble_rest, mass_normalized, rigid_body_basis


@pytest.mark.parametrize("alias", [False, True])
def test_projected_operator_captured_changed_rhs(alias):
    rng = np.random.default_rng(901)
    n = 47
    a = rng.normal(size=(n, n))
    a = a @ a.T + np.eye(n)
    basis = np.linalg.qr(rng.normal(size=(n, 5)))[0].T
    p = np.eye(n) - basis.T @ basis
    op = OrthogonalComplementOperator(SparseOperator(sparse(a)), wp.array(basis, dtype=wp.float64))
    rhs = wp.zeros((3, n), dtype=wp.float64)
    out = rhs if alias else wp.empty_like(rhs)
    op.apply(rhs, out)
    with wp.ScopedCapture() as cap:
        op.apply(rhs, out)
    for _ in range(2):
        b = rng.normal(size=(3, n))
        wp.copy(rhs, wp.array(b, dtype=wp.float64))
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(out.numpy(), (p @ a @ p @ b.T).T, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(out.numpy() @ basis.T, 0, atol=3e-12)


def body():
    x = np.array([[0.0, 0, 0], [1.2, 0, 0], [0, 0.8, 0], [0, 0, 1.1], [1.3, 0.7, 1.2]])
    x += [7.0, -3.0, 2.0]
    h, _, mass = assemble_rest(x, np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), young=10, density=3)
    return x, h, mass


def test_rigid_basis_matches_independent_mass_weighted_nullspace():
    x, h, mass = body()
    q = rigid_body_basis(wp.array(x, dtype=wp.vec3d), mass).numpy()
    raw = []
    for axis in np.eye(3):
        raw.append((np.sqrt(mass.numpy())[:, None] * np.broadcast_to(axis, x.shape)).ravel())
    for axis in np.eye(3):
        raw.append((np.sqrt(mass.numpy())[:, None] * np.cross(axis, x)).ravel())
    expected = np.linalg.qr(np.array(raw).T)[0].T
    np.testing.assert_allclose(q @ q.T, np.eye(6), atol=2e-14)
    np.testing.assert_allclose(q.T @ q, expected.T @ expected, atol=3e-14)
    a = mass_normalized(h, mass)
    out = wp.empty((6, a.shape[0]), dtype=wp.float64)
    SparseOperator(a).apply(wp.array(q, dtype=wp.float64), out)
    np.testing.assert_allclose(out.numpy(), 0, atol=2e-12)


def test_projected_cg_modes_and_captured_initial_vector_updates():
    x, h, mass = body()
    rigid = rigid_body_basis(wp.array(x, dtype=wp.vec3d), mass)
    a = mass_normalized(h, mass)
    n = a.shape[0]
    eye = wp.array(np.eye(n), dtype=wp.float64)
    dense = wp.empty_like(eye)
    SparseOperator(a).apply(eye, dense)
    a_host = dense.numpy().T
    shift = 100.0
    inverse = CGInverse(
        mass_normalized(h, mass, shift), tol=1e-14, maxiter=100, preconditioner="diag"
    )
    op = OrthogonalComplementOperator(inverse, rigid)
    initial = wp.ones(n, dtype=wp.float64)
    solver = KrylovSchur(op, 6, ncv=n, which="LA", initial_vector=initial)
    graph = solver.capture(1)
    for seed in [97, 98]:
        start = np.random.default_rng(seed).normal(size=n)
        wp.copy(initial, wp.array(start, dtype=wp.float64))
        wp.capture_launch(graph)
        vectors = solver.eigenvectors.numpy().T
        values = 1 / solver.eigenvalues.numpy() - shift
        np.testing.assert_allclose(values, np.linalg.eigvalsh(a_host)[6:12], atol=2e-10)
        np.testing.assert_allclose(a_host @ vectors, vectors * values, atol=2e-10)
        np.testing.assert_allclose(rigid.numpy() @ vectors, 0, atol=2e-12)


def test_initial_vector_rejects_wrong_size():
    with pytest.raises(ValueError, match="n FP64"):
        KrylovSchur(
            SparseOperator(sparse(np.eye(12))), 3, initial_vector=wp.ones(11, dtype=wp.float64)
        )
