import numpy as np
import pytest
import warp as wp

from warpack import ComplexCSR, GeneralEigensolver


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("which", ["LM", "LR", "SR"])
def test_general_eigenpairs(complex_input, which):
    n = 80
    k = 5
    rng = np.random.default_rng(123)
    a = rng.normal(size=(n, n)).astype(complex)
    if complex_input:
        a += 1j * rng.normal(size=(n, n))
    op = ComplexCSR(
        n,
        wp.array(np.arange(n + 1) * n, dtype=wp.int32),
        wp.array(np.tile(np.arange(n), n), dtype=wp.int32),
        wp.array(np.stack([a.real.ravel(), a.imag.ravel()], axis=-1), dtype=wp.vec2d),
    )
    s = GeneralEigensolver(op, k, ncv=32, which=which, tol=1.0e-11)
    graph = s.capture(100)
    wp.capture_launch(graph)
    assert s.status.numpy()[0] == 0
    ww = s.eigenvalues.numpy()
    w = ww[:, 0] + 1j * ww[:, 1]
    vv = s.eigenvectors.numpy()
    v = (vv[:, :, 0] + 1j * vv[:, :, 1]).T
    assert s.converged.numpy()[0] == k, s.residuals.numpy()
    assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1.0e-9
    ref = np.linalg.eigvals(a)
    assert max(min(abs(ref - val)) for val in w) < 1.0e-9
    if which == "LM":
        assert min(abs(w)) >= np.sort(abs(ref))[-k] - 1.0e-7
    if which == "LR":
        assert min(w.real) >= np.sort(ref.real)[-k] - 1.0e-7
    if which == "SR":
        assert max(w.real) <= np.sort(ref.real)[k - 1] + 1.0e-7
