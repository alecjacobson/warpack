import numpy as np
import pytest
import warp as wp

from warpack import HermitianCSR, HermitianEigensolver


def complex_operator(a):
    n = len(a)
    return HermitianCSR(
        n,
        wp.array(np.arange(n + 1) * n, dtype=wp.int32),
        wp.array(np.tile(np.arange(n), n), dtype=wp.int32),
        wp.array(np.column_stack((a.real.ravel(), a.imag.ravel())), dtype=wp.vec2d),
    )


def check_pairs(a, solver, expected):
    values = solver.eigenvalues.numpy()
    packed = solver.eigenvectors.numpy()
    vectors = (packed[..., 0] + 1j * packed[..., 1]).T
    scale = max(1.0, np.linalg.norm(a, 2))
    np.testing.assert_allclose(np.sort(values), np.sort(expected), atol=1e-10 * scale, rtol=0)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(solver.k), atol=1e-10)
    assert np.linalg.norm(a @ vectors - vectors * values, axis=0).max() < 1e-10 * scale
    assert solver.count.numpy()[0] == solver.k
    assert solver.real.status.numpy()[0] == 0
    assert solver.real.converged.numpy()[0] == 2 * solver.k


@pytest.mark.parametrize("k", [3, 6, 12])
@pytest.mark.parametrize("which", ["LM", "LA", "SA"])
def test_complex_identity_breakdown(k, which):
    a = np.eye(40, dtype=complex)
    solver = HermitianEigensolver(complex_operator(a), k, which=which, tol=1e-11)
    solver.solve(2)
    check_pairs(a, solver, np.ones(k))
    graph = solver.capture(20)
    for _ in range(2):
        wp.capture_launch(graph)
        check_pairs(a, solver, np.ones(k))


@pytest.mark.parametrize("which", ["LM", "LA", "SA"])
@pytest.mark.parametrize("shift", [0.0, 0.5])
def test_complex_low_rank_breakdown(which, shift):
    rng = np.random.default_rng(0)
    b = rng.normal(size=(100, 5)) + 1j * rng.normal(size=(100, 5))
    a = b @ b.conj().T + shift * np.eye(100)
    expected = np.r_[shift, np.linalg.eigvalsh(b.conj().T @ b) + shift]
    if which == "SA":
        expected = np.full(6, shift)
    solver = HermitianEigensolver(complex_operator(a), 6, which=which, tol=1e-10)
    graph = solver.capture(30)
    for _ in range(2):
        wp.capture_launch(graph)
        check_pairs(a, solver, expected)


@pytest.mark.parametrize("which", ["LM", "LA", "SA", "BE"])
def test_complex_repeated_spectrum_changed_replay(which, tmp_path):
    """A graph must recover after zero/identity without retaining a stale basis."""
    n, k = 40, 3
    rng = np.random.default_rng(45)
    unitary = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    spectrum = np.r_[[-9, -9, -4, -4], np.zeros(n - 8), [7, 7, 12, 12]]
    rotated = (unitary * spectrum) @ unitary.conj().T
    op = complex_operator(np.eye(n, dtype=complex))
    solver = HermitianEigensolver(op, k, ncv=32, which=which, tol=1e-11)
    graph = solver.capture(50)
    for a in [np.zeros((n, n)), rotated, -2 * np.eye(n), rotated]:
        wp.copy(
            op.values, wp.array(np.column_stack((a.real.ravel(), a.imag.ravel())), dtype=wp.vec2d)
        )
        wp.capture_launch(graph)
        ref = np.linalg.eigvalsh(a)
        if which == "LM":
            ref = ref[np.argsort(-abs(ref))[:k]]
        elif which == "LA":
            ref = ref[-k:]
        elif which == "SA":
            ref = ref[:k]
        else:
            ref = np.r_[ref[: k // 2], ref[-(k - k // 2) :]]
        check_pairs(a, solver, ref)
        # An arbitrary basis within a repeated boundary eigenspace is valid.
        values, basis = np.linalg.eigh(a)
        allowed = basis[:, np.min(abs(values[:, None] - ref), axis=1) < 1e-9]
        packed = solver.eigenvectors.numpy()
        vectors = (packed[..., 0] + 1j * packed[..., 1]).T
        assert np.linalg.norm(vectors - allowed @ (allowed.conj().T @ vectors)) < 1e-9
    dot = tmp_path / "hermitian.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = dot.read_text()
    assert "HOST" not in audit and "HtoD" not in audit and "DtoH" not in audit


@pytest.mark.parametrize("ncv", [0, 5, 6])
def test_complex_krylov_requires_expansion_room(ncv):
    with pytest.raises(ValueError):
        HermitianEigensolver(complex_operator(np.eye(10, dtype=complex)), 3, ncv=ncv)


@pytest.mark.parametrize("which", ["LM", "LA", "SA"])
def test_complex_initial_vector_replay(which, tmp_path):
    n, k = 24, 3
    rng = np.random.default_rng(64)
    unitary = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    a = (unitary * np.arange(float(n))) @ unitary.conj().T
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    start = wp.array(np.column_stack((z.real, z.imag)), dtype=wp.vec2d)
    solver = HermitianEigensolver(
        complex_operator(a), k, ncv=24, which=which, tol=1e-10, initial_vector=start
    )
    solver.real.initialize()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.real.initialize()
    graph = solver.capture(40)
    expected = np.arange(3.0) if which == "SA" else np.arange(n - k, n, dtype=float)
    # Start in the exact nullspace, then change both real and imaginary parts,
    # and test safe normalization well outside the squared-norm FP64 range.
    for vector in [1j * unitary[:, 0], z, z * 1e-250, z * 1e250]:
        packed = np.column_stack((vector.real, vector.imag))
        wp.copy(start, wp.array(packed, dtype=wp.vec2d))
        wp.capture_launch(capture.graph)
        scaled = vector / max(abs(vector.real).max(), abs(vector.imag).max())
        normalized = scaled / np.linalg.norm(scaled)
        np.testing.assert_allclose(
            solver.real.q.numpy()[0], np.r_[normalized.real, normalized.imag], atol=1e-14
        )
        wp.capture_launch(graph)
        check_pairs(a, solver, expected)
        np.testing.assert_array_equal(start.numpy(), packed)
    dot = tmp_path / "hermitian-start.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = dot.read_text()
    assert "HOST" not in audit and "HtoD" not in audit and "DtoH" not in audit


def test_complex_invalid_start_recovers_on_replay():
    n = 12
    a = np.diag(np.arange(1.0, n + 1)).astype(complex)
    start = wp.array(np.ones((n, 2)), dtype=wp.vec2d)
    solver = HermitianEigensolver(complex_operator(a), 2, initial_vector=start)
    graph = solver.capture(20)
    for bad in [0.0, float("nan"), float("inf")]:
        invalid = np.zeros((n, 2))
        invalid[0, 1] = bad
        wp.copy(start, wp.array(invalid, dtype=wp.vec2d))
        wp.capture_launch(graph)
        assert solver.real.status.numpy()[0] == 1
        wp.copy(start, wp.array(np.ones((n, 2)), dtype=wp.vec2d))
        wp.capture_launch(graph)
        check_pairs(a, solver, [n - 1, n])


@pytest.mark.parametrize("kind", ["dtype", "length", "ndim", "device"])
def test_complex_start_contract(kind):
    op = complex_operator(np.eye(10, dtype=complex))
    if kind == "dtype":
        start = wp.ones(10, dtype=wp.float64)
    elif kind == "length":
        start = wp.ones(9, dtype=wp.vec2d)
    elif kind == "ndim":
        start = wp.ones((1, 10), dtype=wp.vec2d)
    else:
        start = wp.ones(10, dtype=wp.vec2d, device="cpu")
    with pytest.raises(ValueError, match="initial_vector"):
        HermitianEigensolver(op, 2, initial_vector=start)


@pytest.mark.parametrize("which", ["LM", "LA", "SA"])
def test_complex_hermitian_graph(which):
    rng = np.random.default_rng(24)
    n = 40
    k = 4
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a = a + a.conj().T
    values = np.column_stack((a.real.ravel(), a.imag.ravel()))
    op = HermitianCSR(
        n,
        wp.array(np.arange(n + 1) * n, dtype=wp.int32),
        wp.array(np.tile(np.arange(n), n), dtype=wp.int32),
        wp.array(values, dtype=wp.vec2d),
    )
    s = HermitianEigensolver(op, k, ncv=32, which=which, tol=1.0e-12)
    graph = s.capture(300)
    wp.capture_launch(graph)
    vec = s.eigenvectors.numpy()
    v = (vec[:, :, 0] + 1j * vec[:, :, 1]).T
    w = s.eigenvalues.numpy()
    ref = np.linalg.eigvalsh(a)
    if which == "LM":
        ref = ref[np.argsort(-abs(ref))[:k]]
    elif which == "SA":
        ref = ref[:k]
    else:
        ref = ref[::-1][:k]
    assert s.real.converged.numpy()[0] == 2 * k
    assert s.count.numpy()[0] == k
    np.testing.assert_allclose(w, ref, atol=1.0e-9)
    np.testing.assert_allclose(v.conj().T @ v, np.eye(k), atol=1.0e-9)
    assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1.0e-9
