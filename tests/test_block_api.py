"""Public block-start, mass, constraint and changed-problem contracts."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator


def operator(a):
    return SparseOperator(matrix(sp.csr_matrix(a)))


def assert_pairs(solver, a, mass, expected, wanted=None):
    values, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
    assert solver.status.numpy()[0] == 0
    assert solver.converged.numpy()[0] == solver.k
    np.testing.assert_allclose(values, expected, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(vectors.T @ mass @ vectors, np.eye(solver.k), atol=1e-10)
    assert np.linalg.norm(a @ vectors - (mass @ vectors) * values, axis=0).max() < 1e-7
    if wanted is not None:
        assert la.norm(vectors - wanted @ (wanted.T @ mass @ vectors)) < 1e-7


def test_partial_start_and_native_upstream_preconditioner_replay():
    values = np.arange(1.0, 91.0)
    a = np.diag(values)
    op = operator(a)
    pre = linear.preconditioner(op.matrix, ptype="diag")
    start = wp.empty((2, len(a)), dtype=wp.float64)
    start.assign(np.random.default_rng(51).normal(size=start.shape))
    solver = BlockLOBPCG(op, 4, width=8, preconditioner=pre, initial_vectors=start, tol=1e-10)
    graph = solver.capture(100)
    for seed in (41, 53):
        source = np.random.default_rng(seed).normal(size=start.shape)
        start.assign(source)
        wp.capture_launch(graph)
        assert_pairs(solver, a, np.eye(len(a)), values[:4], np.eye(len(a))[:, :4])
        np.testing.assert_array_equal(start.numpy(), source)


def test_normalized_constraints_and_coupled_mass_refresh_with_partial_start(tmp_path):
    n, k, width, excluded = 64, 4, 8, 2

    def fixture(seed):
        rng = np.random.default_rng(seed)
        lower = np.eye(n) + np.diag(np.linspace(0.05, 0.3 + seed * 0.001, n - 1), -1)
        mass = lower @ lower.T
        q = la.qr(rng.normal(size=(n, n)))[0]
        modes = la.solve_triangular(lower.T, q, lower=False)
        values = np.arange(1.0, n + 1)
        a = mass @ (modes * values) @ modes.T @ mass
        constraints = (modes[:, :excluded] @ np.array([[2.0, 0.4], [0.0, 3.0]])).T.copy()
        return a, mass, modes, values, constraints

    a, mass, modes, values, constraints = fixture(19)
    op, metric, pre = operator(a), operator(mass), operator(la.inv(a + mass))
    basis = wp.array(constraints, dtype=wp.float64)
    starts = wp.array(np.random.default_rng(27).normal(size=(k, n)), dtype=wp.float64)
    solver = BlockLOBPCG(
        op,
        k,
        width=width,
        mass=metric,
        preconditioner=pre,
        initial_vectors=starts,
        constraints=basis,
        normalize_constraints=True,
        tol=1e-10,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
    )
    graph = solver.capture(100)
    for seed in (19, 31):
        a, mass, modes, values, constraints = fixture(seed)
        for target, host in ((op, a), (metric, mass), (pre, la.inv(a + mass))):
            host = sp.csr_matrix(host)
            np.testing.assert_array_equal(target.matrix.columns.numpy(), host.indices)
            target.matrix.values.assign(host.data)
        basis.assign(constraints)
        old_start = starts.numpy().copy()
        wp.capture_launch(graph)
        assert_pairs(solver, a, mass, values[2:6], modes[:, 2:6])
        np.testing.assert_array_equal(basis.numpy(), constraints)
        np.testing.assert_array_equal(starts.numpy(), old_start)
        assert abs(constraints @ mass @ solver.eigenvectors.numpy().T).max() < 1e-9
        wp.copy(starts, solver.eigenvectors)
    dot = tmp_path / "block-api.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert "DtoH" not in text and "HtoD" not in text and "HOST" not in text


@pytest.mark.parametrize("computed", [False, True])
@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
def test_seeded_full_start_explores_modes_after_crossing(metric, repeated, computed):
    n, k, width = 64, 4, 8
    lower = np.eye(n)
    if metric:
        lower += np.diag(np.full(n - 1, 0.2), -1)
    mass = lower @ lower.T
    modes = la.solve_triangular(lower.T, np.eye(n), lower=False)
    old_values = np.arange(1.0, n + 1)
    if repeated:
        old_values[8:] = np.repeat(np.arange(9.0, 23.0), 4)
    # Only old-problem modes enter the start. The new low eigenspace is absent.
    old_a = lower @ np.diag(old_values) @ lower.T
    op, metric_op = operator(old_a), operator(mass)
    if computed:
        source = BlockLOBPCG(
            op,
            width,
            mass=metric_op,
            preconditioner=operator(la.inv(old_a + mass)),
            tol=1e-12,
        )
        wp.capture_launch(source.capture(100))
        assert source.status.numpy()[0] == 0 and source.converged.numpy()[0] == width
        initial = source.x.numpy().copy()
    else:
        initial = modes[:, :width].T.copy()
    start = wp.array(initial, dtype=wp.float64)
    new_values = old_values.copy()
    new_values[:width] += 100
    a = lower @ np.diag(new_values) @ lower.T
    expected, wanted = la.eigh(a, mass, subset_by_index=(0, k - 1))
    solver = BlockLOBPCG(
        op,
        k,
        width=width,
        mass=metric_op,
        preconditioner=operator(la.inv(a + mass)),
        initial_vectors=start,
        initial_perturbation=1e-3,
        tol=1e-10,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
    )
    graph = solver.capture(100)
    wp.capture_launch(graph)
    assert_pairs(solver, old_a, mass, old_values[:k], modes[:, :k])
    current = sp.csr_matrix(a)
    np.testing.assert_array_equal(op.matrix.columns.numpy(), current.indices)
    op.matrix.values.assign(current.data)
    for _ in range(2):
        wp.capture_launch(graph)
        assert_pairs(solver, a, mass, expected, wanted)
        np.testing.assert_array_equal(start.numpy(), initial)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", ["dependent", "nan"])
def test_invalid_constraint_basis_cannot_report_convergence(normalize, kind):
    constraints = np.eye(24)[:2].copy()
    if kind == "dependent":
        constraints[1] = constraints[0]
    else:
        constraints[0, 0] = np.nan
    solver = BlockLOBPCG(
        operator(np.eye(24)),
        2,
        width=4,
        constraints=wp.array(constraints, dtype=wp.float64),
        normalize_constraints=normalize,
    )
    wp.capture_launch(solver.capture(5))
    assert solver.status.numpy()[0] != 0
    assert solver.converged.numpy()[0] == 0


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf")])
def test_invalid_perturbation_rejected(amount):
    with pytest.raises(ValueError, match="initial_perturbation"):
        BlockLOBPCG(operator(np.eye(24)), 2, initial_perturbation=amount)


def test_block_metadata_rejected_before_iteration():
    op = operator(np.eye(24))
    with pytest.raises(ValueError, match="width"):
        BlockLOBPCG(op, 2, width=0)
    with pytest.raises(ValueError, match="preconditioner"):
        BlockLOBPCG(op, 2, preconditioner=operator(np.eye(12)))
    with pytest.raises(ValueError, match="initial_vectors"):
        BlockLOBPCG(op, 2, width=4, initial_vectors=wp.zeros((5, 24), dtype=wp.float64))
    solver = BlockLOBPCG(op, 2, width=4)
    with pytest.raises(ValueError, match="max_iterations"):
        solver.capture(0)
    with pytest.raises(ValueError, match="steps"):
        solver.solve(-1)


def test_high_fiedler_start_returns_requested_low_spectrum():
    # Upstream's n=12 fixture compares SciPy/CuPy agreement even though both
    # retain the high invariant start. Assert the analytic requested pair.
    n = 12
    adjacency = la.toeplitz(np.r_[0.0, 1.0, np.zeros(n - 2)])
    a = np.diag(adjacency.sum(axis=1)) - adjacency
    basis = np.cos(np.outer(np.arange(n) + 0.5, np.pi * np.arange(n) / n))
    source = basis[:, -2:].T.copy()
    solver = BlockLOBPCG(
        operator(a),
        2,
        initial_vectors=wp.array(source, dtype=wp.float64),
        initial_perturbation=1e-3,
        tol=1e-10,
    )
    graph = solver.capture(100)
    wp.capture_launch(graph)
    wanted = la.orth(basis[:, :2])
    assert_pairs(solver, a, np.eye(n), 2 - 2 * np.cos(np.pi * np.arange(2) / n), wanted)
