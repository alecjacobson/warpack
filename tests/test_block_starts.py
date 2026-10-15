"""Caller-owned starting subspaces remain device inputs during graph replay."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import GeneralizedEigensolver, SparseOperator, SymmetricEigensolver


def operator(a):
    return SparseOperator(matrix(sp.csr_matrix(a)))


@pytest.mark.parametrize("generalized", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
def test_changed_block_and_matrix_replay(generalized, repeated):
    n, k, m = 32, 4, 8
    rng = np.random.default_rng(179)
    mass = np.eye(n)
    if generalized:
        lower = np.eye(n) + np.diag(np.full(n - 1, 0.1), -1)
        mass = lower @ lower.T
    else:
        lower = np.eye(n)
    rotation = la.qr(rng.normal(size=(n, n)))[0]
    vectors = la.solve_triangular(lower.T, rotation, lower=False)
    eigenvalues = np.arange(1.0, n + 1)
    if repeated:
        eigenvalues = np.ceil(eigenvalues / 2)
    a = mass @ (vectors * eigenvalues) @ vectors.T @ mass
    op = operator(a)
    inverse = operator(la.inv(a))
    # Full blocks make the supplied span observable with only one step; random
    # initialization could not reproduce these exact answers at this budget.
    start = wp.array(vectors[:, :m].T.copy(), dtype=wp.float64, device=op.device)
    if generalized:
        solver = GeneralizedEigensolver(
            op, operator(mass), inverse, k, ncv=m, initial_vectors=start
        )
    else:
        solver = SymmetricEigensolver(
            op, k, ncv=m, iteration_operator=inverse, initial_vectors=start
        )
    graph = solver.capture(1, adaptive=False)
    for order in (np.arange(n), np.roll(np.arange(n), 13), np.arange(n)):
        current = vectors[:, order]
        a = mass @ (current * eigenvalues) @ current.T @ mass
        # All entries are present; update values without changing device pointers.
        wp.copy(op.matrix.values, operator(a).matrix.values)
        wp.copy(inverse.matrix.values, operator(la.inv(a)).matrix.values)
        supplied = current[:, :m].T.copy()
        start.assign(supplied)
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == k
        np.testing.assert_allclose(w, eigenvalues[:k], atol=2e-11)
        np.testing.assert_allclose(a @ v, (mass @ v) * w, atol=2e-10)
        np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), atol=2e-12)
        np.testing.assert_array_equal(start.numpy(), supplied)


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_partial_block_gets_independent_padding(rows):
    n, k, m = 32, 4, 8
    mass = np.diag(np.linspace(1, 2, n))
    mass += np.diag(np.full(n - 1, 0.1), 1) + np.diag(np.full(n - 1, 0.1), -1)
    a = np.diag(np.arange(1.0, n + 1))
    start = wp.array(np.eye(n)[:rows], dtype=wp.float64)
    solver = GeneralizedEigensolver(
        operator(a),
        operator(mass),
        operator(la.inv(a)),
        k,
        ncv=m,
        initial_vectors=start,
    )
    solver.initialize()
    q = solver.q.numpy()
    np.testing.assert_allclose(q @ mass @ q.T, np.eye(m), atol=2e-12)
    np.testing.assert_allclose(q[:rows, rows:], 0, atol=0)
    wp.capture_launch(solver.capture(100, adaptive=True))
    assert solver.status.numpy()[0] == 0
    assert solver.converged.numpy()[0] == k
    np.testing.assert_allclose(
        solver.eigenvalues.numpy(), la.eigh(a, mass, eigvals_only=True)[:k], atol=1e-9
    )


@pytest.mark.parametrize("generalized", [False, True])
@pytest.mark.parametrize("invalid", ["dependent", "nan", "inf"])
def test_invalid_block_reports_device_failure(generalized, invalid):
    op = operator(np.eye(16))
    block = np.eye(16)[:4].copy()
    if invalid == "dependent":
        block[1] = block[0]
    else:
        block[0, 0] = float(invalid)
    start = wp.array(block, dtype=wp.float64)
    solver = (
        GeneralizedEigensolver(op, op, op, 2, ncv=4, initial_vectors=start)
        if generalized
        else SymmetricEigensolver(op, 2, ncv=4, which="LM", initial_vectors=start)
    )
    solver.initialize()
    with wp.ScopedCapture() as capture:
        solver.initialize()
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] != 0


@pytest.mark.parametrize(
    "shape,dtype,device",
    [
        ((32,), wp.float64, "cuda:0"),
        ((0, 32), wp.float64, "cuda:0"),
        ((9, 32), wp.float64, "cuda:0"),
        ((4, 31), wp.float64, "cuda:0"),
        ((4, 32), wp.float32, "cuda:0"),
        ((4, 32), wp.float64, "cpu"),
    ],
)
def test_block_metadata_validation(shape, dtype, device):
    op = operator(np.eye(32))
    start = wp.zeros(shape, dtype=dtype, device=device)
    with pytest.raises(ValueError, match="initial_vectors"):
        SymmetricEigensolver(op, 4, ncv=8, which="LM", initial_vectors=start)
