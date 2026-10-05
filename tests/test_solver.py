import numpy as np
import pytest
import warp as wp
import warp.sparse as ws

from warpack import CuDSSInverse, SparseOperator, SymmetricEigensolver


def sparse(a, device="cuda:0"):
    r, c = np.nonzero(a)
    return ws.bsr_from_triplets(
        a.shape[0],
        a.shape[1],
        wp.array(r, dtype=wp.int32, device=device),
        wp.array(c, dtype=wp.int32, device=device),
        wp.array(a[r, c], dtype=wp.float64, device=device),
    )


@pytest.mark.parametrize("n,k", [(10, 3), (100, 10)])
def test_dense_full_space(n, k):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(n, n))
    a += a.T
    solver = SymmetricEigensolver(SparseOperator(sparse(a)), k, ncv=n)
    solver.solve(iterations=0)
    vals, vec = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
    np.testing.assert_allclose(vals, np.linalg.eigvalsh(a)[:k], atol=1.0e-10)
    assert np.max(np.abs(a @ vec - vec * vals)) < 1.0e-9
    np.testing.assert_allclose(vec.T @ vec, np.eye(k), atol=1.0e-11)


def test_graph_replay():
    a = np.diag(np.arange(1.0, 101.0))
    op = SparseOperator(sparse(a))
    solver = SymmetricEigensolver(op, 5, ncv=16, which="LA", iteration_operator=op)
    solver.solve(1)
    with wp.ScopedCapture(device="cuda:0") as cap:
        solver.solve(150)
    for _ in range(2):
        wp.capture_launch(cap.graph)
        np.testing.assert_allclose(
            solver.eigenvalues.numpy(), np.arange(100.0, 95.0, -1), atol=1.0e-9
        )
        assert solver.residuals.numpy().max() < 1.0e-9


def test_inverse_graph():
    n = 100
    a = (
        np.diag(np.arange(1.0, n + 1))
        + np.diag(np.full(n - 1, 0.1), 1)
        + np.diag(np.full(n - 1, 0.1), -1)
    )
    mat = sparse(a)
    inverse = CuDSSInverse(mat, 16)
    solver = SymmetricEigensolver(SparseOperator(mat), 5, ncv=16, iteration_operator=inverse)
    solver.solve(1)
    with wp.ScopedCapture(device="cuda:0") as cap:
        solver.solve(30)
    wp.capture_launch(cap.graph)
    np.testing.assert_allclose(solver.eigenvalues.numpy(), np.linalg.eigvalsh(a)[:5], atol=1.0e-10)
    assert solver.residuals.numpy().max() < 1.0e-9
    inverse.close()


@pytest.mark.parametrize("which", ["LM", "SA", "LA"])
def test_krylov(which):
    from warpack import KrylovSchur

    rng = np.random.default_rng(12)
    a = rng.normal(size=(100, 100))
    a += a.T
    s = KrylovSchur(SparseOperator(sparse(a)), 5, ncv=32, which=which)
    s.solve(30)
    vals, vec = s.eigenvalues.numpy(), s.eigenvectors.numpy().T
    ref = np.linalg.eigvalsh(a)
    if which == "LM":
        ref = ref[np.argsort(-np.abs(ref))[:5]]
    elif which == "SA":
        ref = ref[:5]
    else:
        ref = ref[::-1][:5]
    np.testing.assert_allclose(vals, ref, atol=1.0e-9)
    assert np.max(np.abs(a @ vec - vec * vals)) < 1.0e-8


def test_device_convergence_loop():
    from warpack import KrylovSchur

    rng = np.random.default_rng(12)
    a = rng.normal(size=(100, 100))
    a += a.T
    s = KrylovSchur(SparseOperator(sparse(a)), 5, ncv=32, which="LM", tol=1.0e-10)
    graph = s.capture(100)
    for _ in range(2):
        wp.capture_launch(graph)
        assert s.converged.numpy()[0] == 5
        assert s.iterations.numpy()[0] < 100
        assert s.residuals.numpy().max() < 1.0e-10
        v = s.eigenvectors.numpy().T
        w = s.eigenvalues.numpy()
        assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1.0e-8


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("diag", [np.ones(80), np.zeros(80), np.repeat(np.arange(1.0, 9.0), 10)])
def test_repeated_and_zero_spectrum(diag, captured):
    from warpack import KrylovSchur

    s = KrylovSchur(SparseOperator(sparse(np.diag(diag))), 6, ncv=32, which="LA", tol=1.0e-10)
    if captured:
        wp.capture_launch(s.capture(8))
    else:
        s.solve(8)
    w = s.eigenvalues.numpy()
    v = s.eigenvectors.numpy().T
    np.testing.assert_allclose(w, np.sort(diag)[-6:][::-1], atol=1.0e-9)
    np.testing.assert_allclose(v.T @ v, np.eye(6), atol=1.0e-10)
    assert np.linalg.norm(diag[:, None] * v - v * w, axis=0).max() < 1.0e-9


def test_generalized_nondiagonal_mass():
    import scipy.linalg as la

    from warpack import GeneralizedEigensolver

    n = 50
    a = (
        np.diag(np.arange(1.0, n + 1))
        + np.diag(np.full(n - 1, 0.2), 1)
        + np.diag(np.full(n - 1, 0.2), -1)
    )
    b = (
        np.diag(np.linspace(1.0, 2.0, n))
        + np.diag(np.full(n - 1, 0.1), 1)
        + np.diag(np.full(n - 1, 0.1), -1)
    )
    wa, wb = sparse(a), sparse(b)
    inv = CuDSSInverse(wa, 16)
    s = GeneralizedEigensolver(SparseOperator(wa), SparseOperator(wb), inv, 5, ncv=16)
    graph = s.capture(35, adaptive=False)
    wp.capture_launch(graph)
    v = s.eigenvectors.numpy().T
    w = s.eigenvalues.numpy()
    np.testing.assert_allclose(w, la.eigh(a, b, eigvals_only=True)[:5], atol=1.0e-9)
    np.testing.assert_allclose(v.T @ b @ v, np.eye(5), atol=1.0e-10)
    assert np.linalg.norm(a @ v - b @ v * w, axis=0).max() < 1.0e-8
    inv.close()


def test_pure_warp_cg_inverse():
    from warpack import CGInverse, KrylovSchur

    n = 60
    a = (
        np.diag(np.arange(1.0, n + 1))
        + np.diag(np.full(n - 1, 0.1), 1)
        + np.diag(np.full(n - 1, 0.1), -1)
    )
    inv = CGInverse(sparse(a), tol=1.0e-13, maxiter=100)
    s = KrylovSchur(inv, 5, ncv=24, which="LA", tol=1.0e-10)
    graph = s.capture(10)
    wp.capture_launch(graph)
    v = s.eigenvectors.numpy().T
    w = 1 / s.eigenvalues.numpy()
    np.testing.assert_allclose(w, np.linalg.eigvalsh(a)[:5], atol=1.0e-9)
    assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1.0e-8


def test_both_ends():
    from warpack import KrylovSchur

    rng = np.random.default_rng(24)
    a = rng.normal(size=(70, 70))
    a += a.T
    s = KrylovSchur(SparseOperator(sparse(a)), 6, ncv=32, which="BE", tol=1.0e-10)
    wp.capture_launch(s.capture(100))
    w = s.eigenvalues.numpy()
    ref = np.linalg.eigvalsh(a)
    np.testing.assert_allclose(np.sort(w), np.r_[ref[:3], ref[-3:]], atol=1.0e-9)


def test_generalized_interior_shift():
    import scipy.linalg as la

    from warpack import GeneralizedEigensolver

    a = np.diag(np.arange(1.0, 41.0))
    b = np.diag(np.linspace(1.0, 1.5, 40))
    sigma = 7.3
    wa, wb = sparse(a), sparse(b)
    inv = CuDSSInverse(sparse(a - sigma * b), 16, mtype="symmetric")
    s = GeneralizedEigensolver(SparseOperator(wa), SparseOperator(wb), inv, 4, ncv=16, target=sigma)
    wp.capture_launch(s.capture(40, adaptive=False))
    ref = la.eigh(a, b, eigvals_only=True)
    ref = ref[np.argsort(abs(ref - sigma))[:4]]
    np.testing.assert_allclose(s.eigenvalues.numpy(), ref, atol=1.0e-9)
    assert s.residuals.numpy().max() < 1.0e-9
    inv.close()


def test_iteration_budget_reports_nonconvergence():
    from warpack import KrylovSchur

    rng = np.random.default_rng(34)
    a = rng.normal(size=(100, 100))
    a += a.T
    s = KrylovSchur(SparseOperator(sparse(a)), 4, ncv=12, tol=1.0e-12)
    wp.capture_launch(s.capture(1))
    assert s.iterations.numpy()[0] == 1
    assert s.converged.numpy()[0] < 4


def test_graph_observes_changed_matrix_values():
    from warpack import KrylovSchur

    rng = np.random.default_rng(34)
    a = rng.normal(size=(40, 40))
    a += a.T
    mat = sparse(a)
    s = KrylovSchur(SparseOperator(mat), 4, ncv=24, tol=1.0e-11)
    graph = s.capture(100)
    wp.capture_launch(graph)
    first = s.eigenvalues.numpy()
    wp.copy(mat.values, wp.array(mat.values.numpy() * 2, dtype=wp.float64))
    wp.capture_launch(graph)
    np.testing.assert_allclose(s.eigenvalues.numpy(), 2 * first, atol=1.0e-9)


def test_fractional_shift_preserves_double_precision():
    a = np.diag(np.arange(1.0, 6.0))
    shift, sign = 0.123456789012345, 0.987654321098765
    op = SparseOperator(sparse(a), shift=shift, sign=sign)
    x = wp.ones((1, 5), dtype=wp.float64, device="cuda:0")
    y = wp.empty_like(x)
    op.apply(x, y)
    np.testing.assert_allclose(y.numpy()[0], sign * np.diag(a) + shift, rtol=1e-15)


def test_incremental_projection_matches_original_operator():
    from warpack import KrylovSchur

    rng = np.random.default_rng(921)
    a = rng.normal(size=(120, 120))
    a += a.T

    class CheckedProjection(KrylovSchur):
        def diagonalize(self):
            q = self.q.numpy()
            np.testing.assert_allclose(self.g.numpy(), q @ a @ q.T, atol=2e-11)
            super().diagonalize()

    solver = CheckedProjection(SparseOperator(sparse(a)), 7, ncv=32, tol=1e-11)
    solver.solve(8)
    assert solver.residuals.numpy().max() < 1e-9


def test_pure_warp_graph_has_no_host_nodes_or_transfers(tmp_path):
    from warpack import KrylovSchur

    a = np.diag(np.arange(1.0, 81.0))
    solver = KrylovSchur(SparseOperator(sparse(a)), 5, ncv=24)
    graph = solver.capture(10)
    path = tmp_path / "graph.dot"
    wp.capture_debug_dot_print(graph, str(path))
    text = path.read_text()
    assert "HtoD" not in text and "DtoH" not in text
    assert 'label="{HOST' not in text
    assert "CONDITIONAL" in text
