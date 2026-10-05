import numpy as np
import pytest
import warp as wp

from warpack import ComplexCSR, ShiftInvertEigensolver


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
