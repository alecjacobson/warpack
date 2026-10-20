"""Independent checks for the experimental, unexported pure-Warp block method."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.experimental.block_lobpcg import BlockLOBPCG
from benchmarks.upstream.adapters import matrix
from warpack import CGInverse, SparseOperator


def operator(a):
    return SparseOperator(matrix(sp.csr_matrix(a)))


@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("kind", ["spd", "indefinite", "repeated", "rigid"])
@pytest.mark.parametrize("optimized", [False, True])
def test_original_pencil_and_requested_subspace(metric, kind, optimized):
    n, k, width = 64, 4, 8
    rng = np.random.default_rng(61)
    lower = np.eye(n)
    if metric:
        lower += np.diag(np.linspace(0.05, 0.4, n - 1), -1)
    mass = lower @ lower.T
    q = la.qr(rng.normal(size=(n, n)))[0]
    modes = la.solve_triangular(lower.T, q, lower=False)
    values = np.arange(1.0, n + 1)
    if kind == "indefinite":
        values -= 4
    if kind in ("repeated", "rigid"):
        values = np.ceil(values / 2)
    excluded = 6 if kind == "rigid" else 0
    values[:excluded] = 0
    a = mass @ (modes * values) @ modes.T @ mass
    constraints = None
    if excluded:
        constraints = wp.array(modes[:, :excluded].T.copy(), dtype=wp.float64)
    initial = rng.normal(size=(width, n))
    starts = wp.array(initial, dtype=wp.float64)
    s = BlockLOBPCG(
        operator(a),
        k,
        width,
        mass=operator(mass),
        preconditioner=operator(la.inv(a + 5 * mass)),
        initial_vectors=starts,
        constraints=constraints,
        symmetric_products=optimized,
        fused_updates=optimized,
        paired_rotation=optimized,
        tol=1e-10,
    )
    graph = s.capture(100)
    for seed in (17, 29):
        initial = np.random.default_rng(seed).normal(size=(width, n))
        starts.assign(initial)
        wp.capture_launch(graph)
        w, x = s.eigenvalues.numpy(), s.eigenvectors.numpy().T
        assert s.status.numpy()[0] == 0
        assert s.converged.numpy()[0] == k
        assert s.iterations.numpy()[0] < 100
        np.testing.assert_allclose(w, values[excluded : excluded + k], atol=1e-9)
        np.testing.assert_allclose(x.T @ mass @ x, np.eye(k), atol=1e-10)
        residual = la.norm(a @ x - (mass @ x) * w, axis=0)
        assert residual.max() < 1e-8
        wanted = modes[:, excluded : excluded + k]
        assert la.norm(x - wanted @ (wanted.T @ mass @ x)) < 1e-8
        np.testing.assert_array_equal(starts.numpy(), initial)
        if excluded:
            assert abs(modes[:, :excluded].T @ mass @ x).max() < 1e-10


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("optimized", [False, True])
def test_rank_deficient_search_at_exact_convergence(value, optimized):
    op = operator(np.eye(48) * value)
    s = BlockLOBPCG(
        op,
        4,
        width=8,
        symmetric_products=optimized,
        fused_updates=optimized,
        paired_rotation=optimized,
        tol=1e-12,
    )
    # Keep stepping after exact convergence: zero/dependent residual and history
    # blocks must not destroy the full-rank current eigenvector block.
    s.solve(6)
    x = s.eigenvectors.numpy()
    assert s.status.numpy()[0] == 0
    assert s.converged.numpy()[0] == 4
    np.testing.assert_allclose(
        s.eigenvalues.numpy(),
        value,
        atol=1e-12,
    )
    np.testing.assert_allclose(x @ x.T, np.eye(4), atol=1e-11)


@pytest.mark.parametrize("bad", ["zero", "nan", "inf"])
def test_invalid_initial_block_never_reports_convergence(bad):
    start = np.zeros((4, 24))
    if bad != "zero":
        start[0, 0] = float(bad)
    s = BlockLOBPCG(
        operator(np.eye(24)), 2, width=4, initial_vectors=wp.array(start, dtype=wp.float64)
    )
    wp.capture_launch(s.capture(10))
    assert s.status.numpy()[0] != 0
    assert s.converged.numpy()[0] == 0


@pytest.mark.parametrize("optimized", [False, True])
def test_wide_block_preserves_accurate_modes_after_convergence(optimized):
    # A first formulation whitened [X,W,P] together, amplifying roundoff in X
    # when W/P became dependent. It degraded from 1e-9 to 1e-5 residuals here.
    n = 100
    q = la.qr(np.random.default_rng(83).normal(size=(n, n)))[0]
    a = (q * np.arange(1.0, n + 1)) @ q.T
    s = BlockLOBPCG(
        operator(a),
        4,
        width=24,
        preconditioner=operator(la.inv(a)),
        symmetric_products=optimized,
        fused_updates=optimized,
        paired_rotation=optimized,
        tol=1e-10,
    )
    s.solve(10)
    for _ in range(20):
        s.step()
        assert s.status.numpy()[0] == 0
        assert s.residuals.numpy().max() < 1e-10
        x = s.eigenvectors.numpy()
        np.testing.assert_allclose(x @ x.T, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(s.eigenvalues.numpy(), [1.0, 2.0, 3.0, 4.0], atol=1e-11)


def test_truncated_cg_preconditioning_in_static_graph():
    a = np.diag(np.linspace(1, 30, 64))
    a += np.diag(np.full(63, 0.4), 1) + np.diag(np.full(63, 0.4), -1)
    device = matrix(sp.csr_matrix(a))
    pre = CGInverse(device, tol=0.01, atol=0.0, maxiter=5)
    s = BlockLOBPCG(SparseOperator(device), 4, width=8, preconditioner=pre, tol=1e-9)
    with pytest.raises(ValueError, match="adaptive=False"):
        s.capture(30, adaptive=True)
    graph = s.capture(30)
    for _ in range(2):
        wp.capture_launch(graph)
        assert s.status.numpy()[0] == 0
        assert s.converged.numpy()[0] == 4
        np.testing.assert_allclose(s.eigenvalues.numpy(), la.eigvalsh(a)[:4], atol=1e-10)
        stats = pre.stats.numpy()
        assert stats[3] == 30 * 8
        assert stats[1] <= 5


@pytest.mark.parametrize("optimized", [False, True])
def test_changed_pencil_constraints_and_start_reuse_one_graph(optimized):
    # Dense coupled fixtures preserve the sparse structure while every numerical
    # input changes. Both excluded and requested invariant spaces move.
    n, k, width, excluded = 48, 4, 8, 6

    def fixture(seed):
        rng = np.random.default_rng(seed)
        lower = np.eye(n) + np.tril(rng.normal(scale=0.03, size=(n, n)), -1)
        mass = lower @ lower.T
        q = la.qr(rng.normal(size=(n, n)))[0]
        modes = la.solve_triangular(lower.T, q, lower=False)
        values = np.arange(n, dtype=float)
        values[:excluded] = 0
        a = mass @ (modes * values) @ modes.T @ mass
        return a, mass, modes, values

    a, mass, modes, values = fixture(21)
    op, metric, pre = operator(a), operator(mass), operator(la.inv(a + mass))
    constraints = wp.array(modes[:, :excluded].T.copy(), dtype=wp.float64)
    starts = wp.array(np.random.default_rng(2).normal(size=(width, n)), dtype=wp.float64)
    s = BlockLOBPCG(
        op,
        k,
        width,
        mass=metric,
        preconditioner=pre,
        constraints=constraints,
        initial_vectors=starts,
        symmetric_products=optimized,
        fused_updates=optimized,
        paired_rotation=optimized,
        tol=1e-10,
    )
    graph = s.capture(100)
    for seed in (21, 22):
        a, mass, modes, values = fixture(seed)
        for target, host in ((op, a), (metric, mass), (pre, la.inv(a + mass))):
            host_csr = sp.csr_matrix(host)
            np.testing.assert_array_equal(target.matrix.columns.numpy(), host_csr.indices)
            target.matrix.values.assign(host_csr.data)
        constraints.assign(modes[:, :excluded].T.copy())
        wp.capture_launch(graph)
        x, w = s.eigenvectors.numpy().T, s.eigenvalues.numpy()
        assert s.status.numpy()[0] == 0 and s.converged.numpy()[0] == k
        np.testing.assert_allclose(w, values[excluded : excluded + k], atol=1e-9)
        assert la.norm(a @ x - (mass @ x) * w) < 1e-8
        np.testing.assert_allclose(x.T @ mass @ x, np.eye(k), atol=1e-10)
        wanted = modes[:, excluded : excluded + k]
        assert la.norm(x - wanted @ (wanted.T @ mass @ x)) < 1e-8
        assert abs(constraints.numpy() @ mass @ x).max() < 1e-10
        # Reuse the computed oversampled block as the next problem's start.
        wp.copy(starts, s.x)


@pytest.mark.parametrize("coupled_mass", [False, True])
def test_wide_block_captured_repeated_modes(coupled_mass, tmp_path):
    # Width 24 exercises the 72-column projected solve, including rank loss
    # after converging repeated modes. Continue a fixed budget after convergence.
    n, k = 100, 4
    rng = np.random.default_rng(836)
    q = la.qr(rng.normal(size=(n, n)))[0]
    values = np.arange(1.0, n + 1)
    values[:4] = [1, 1, 2, 2]
    coordinates = np.eye(n)
    if coupled_mass:
        coordinates += np.tril(rng.normal(scale=0.02, size=(n, n)), -1)
    mass = coordinates.T @ coordinates
    a = coordinates.T @ ((q * values) @ q.T) @ coordinates
    requested = la.solve(coordinates, q[:, :k])
    solver = BlockLOBPCG(
        operator(a),
        k,
        width=24,
        mass=operator(mass) if coupled_mass else None,
        preconditioner=operator(la.inv(a)),
        tol=1e-10,
    )
    graph = solver.capture(30, adaptive=False)
    for _ in range(2):
        wp.capture_launch(graph)
        w, x = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == k
        np.testing.assert_allclose(np.sort(w), values[:k], rtol=0, atol=1e-10)
        np.testing.assert_allclose(x.T @ mass @ x, np.eye(k), rtol=0, atol=1e-10)
        av, mv = a @ x, mass @ x
        residual = la.norm(av - mv * w, axis=0) / np.maximum(
            1, la.norm(av, axis=0) + abs(w) * la.norm(mv, axis=0)
        )
        assert residual.max() < 1e-10
        remainder = x - requested @ (requested.T @ mv)
        assert np.sqrt(np.sum(remainder * (mass @ remainder))) < 1e-8
    dot = tmp_path / "wide-block.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(word in dot.read_text() for word in ("HOST", "HtoD", "DtoH"))
