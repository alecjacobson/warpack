import numpy as np
import pytest
import warp as wp

from warpack import HermitianCSR, HermitianEigensolver


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
