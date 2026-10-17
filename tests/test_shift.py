import re

import numpy as np
import pytest
import warp as wp

from warpack import ComplexCSR, ShiftInvertEigensolver


def check_hermitian_pairs(a, solver, sigma):
    from benchmarks.upstream.adapters import output

    w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
    ref, basis = np.linalg.eigh(a)
    selected = ref[np.argsort(abs(ref - sigma))[: solver.k]]
    np.testing.assert_allclose(np.sort(w), np.sort(selected), atol=1e-9, rtol=0)
    assert np.max(abs(a @ v - v * w)) < 1e-9
    np.testing.assert_allclose(v.conj().T @ v, np.eye(solver.k), atol=1e-10, rtol=0)
    allowed = basis[:, np.min(abs(ref[:, None] - selected), axis=1) < 1e-9]
    assert np.linalg.norm(v - allowed @ (allowed.conj().T @ v), 2) < 1e-9
    assert solver.converged.numpy()[0] == solver.k
    assert solver.status.numpy()[0] == 0


@pytest.mark.parametrize("n,k,ncv", [(10, 3, 6), (100, 10, 20)])
def test_spectra_sparse_hermitian_shift(n, k, ncv):
    from pathlib import Path

    from benchmarks.upstream.adapters import complex_matrix

    a = np.load(Path(__file__).parent / "fixtures/spectra_hermitian_sparse.npz")[f"a{n}"]
    solver = ShiftInvertEigensolver(complex_matrix(a, True), k, ncv=ncv, hermitian=True, tol=1e-12)
    graph = solver.capture(30)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            check_hermitian_pairs(a, solver, 0.0)
    finally:
        solver.graph = solver.core.real.graph = graph = None
        solver.close()


@pytest.mark.parametrize("backend", ["cudss", "gmres"])
def test_hermitian_shift_changed_start_and_repeated_modes(backend, tmp_path):
    from benchmarks.upstream.adapters import complex_matrix

    n, sigma = 12, 0.1
    rng = np.random.default_rng(92)
    u = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    a = (u * [-6, -4, -2, 0, 0, 1, 1, 3, 5, 7, 9, 11]) @ u.conj().T
    start = wp.array(np.column_stack((u[:, 0].real, u[:, 0].imag)), dtype=wp.vec2d)
    solver = ShiftInvertEigensolver(
        complex_matrix(a),
        3,
        sigma,
        ncv=n,
        backend=backend,
        hermitian=True,
        initial_vector=start,
        tol=1e-10,
        inner_tol=1e-13,
        inner_maxiter=100,
    )
    graph = solver.capture(2)
    try:
        for vector in [u[:, 0], 1j * u[:, -1]]:
            packed = np.column_stack((vector.real, vector.imag))
            wp.copy(start, wp.array(packed, dtype=wp.vec2d))
            wp.capture_launch(graph)
            check_hermitian_pairs(a, solver, sigma)
            np.testing.assert_array_equal(start.numpy(), packed)
        dot = tmp_path / "hermitian-shift.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        audit = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', audit) and "DtoH" not in audit
        if backend == "gmres":
            assert "HtoD" not in audit
        # cuDSS's vendor HtoD scalar copies remain tracked in issue #29.
    finally:
        solver.graph = solver.core.real.graph = graph = None
        solver.close()


@pytest.mark.parametrize("options", [dict(hermitian=True, sigma=1j), dict(initial_vector=True)])
def test_shift_rejects_incompatible_hermitian_arguments(options):
    with pytest.raises(ValueError):
        ShiftInvertEigensolver(None, 2, **options)


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("sigma", [0.0, 0.2 + 0.3j])
def test_complex_sparse_inverse_original_residual(in_place, sigma):
    from pathlib import Path

    from benchmarks.upstream.adapters import complex_matrix
    from warpack import ComplexShiftInverse

    a = np.load(Path(__file__).parent / "fixtures/spectra_hermitian_sparse.npz")["a10"]
    shifted = a - sigma * np.eye(len(a))
    real = np.block([[shifted.real, -shifted.imag], [shifted.imag, shifted.real]])
    inverse = ComplexShiftInverse(complex_matrix(a), sigma)
    rhs = wp.zeros((2, len(real)), dtype=wp.float64)
    out = rhs if in_place else wp.empty_like(rhs)
    inverse.apply(rhs, out)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        inverse.apply(rhs, out)
    try:
        for seed in [1, 2]:
            b = np.random.default_rng(seed).normal(size=rhs.shape)
            wp.copy(rhs, wp.array(b, dtype=wp.float64))
            wp.capture_launch(capture.graph)
            x = out.numpy()
            assert np.linalg.norm(x @ real.T - b) / np.linalg.norm(b) < 1e-13
            np.testing.assert_allclose(x, np.linalg.solve(real, b.T).T, atol=1e-12, rtol=1e-12)
    finally:
        capture.graph = None
        inverse.close()


@pytest.mark.parametrize("backend", ["cudss", "gmres"])
@pytest.mark.parametrize("sigma", [3.2, 3.2 + 0.4j])
def test_complex_shift(backend, sigma):
    n = 20
    k = 3
    rng = np.random.default_rng(12)
    a = np.diag(np.arange(1.0, n + 1)).astype(complex) + 0.1 * (
        rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    )
    op = ComplexCSR(
        n,
        wp.array(np.arange(n + 1) * n, dtype=wp.int32),
        wp.array(np.tile(np.arange(n), n), dtype=wp.int32),
        wp.array(np.stack([a.real.ravel(), a.imag.ravel()], -1), dtype=wp.vec2d),
    )
    s = ShiftInvertEigensolver(
        op,
        k,
        sigma,
        ncv=16,
        backend=backend,
        tol=1.0e-9,
        inner_tol=1.0e-13,
        inner_maxiter=200,
    )
    graph = s.capture(4)
    wp.capture_launch(graph)
    ww = s.eigenvalues.numpy()
    w = ww[:, 0] + 1j * ww[:, 1]
    vv = s.eigenvectors.numpy()
    v = (vv[:, :, 0] + 1j * vv[:, :, 1]).T
    ref = np.linalg.eigvals(a)
    ref = ref[np.argsort(abs(ref - sigma))[:k]]
    assert max(min(abs(ref - val)) for val in w) < 1.0e-8
    assert s.converged.numpy()[0] == k, s.residuals.numpy()
    assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1.0e-8
    s.close()
