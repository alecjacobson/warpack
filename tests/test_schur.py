import numpy as np
import pytest
import warp as wp

from warpack.schur import schur


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("n", [4, 16, 40])
def test_complex_schur(n, complex_input):
    rng = np.random.default_rng(81)
    a = rng.normal(size=(n, n)).astype(complex)
    if complex_input:
        a += 1j * rng.normal(size=(n, n))
    data = np.stack([a.real, a.imag], axis=-1)
    h = wp.array(data, dtype=wp.vec2d)
    q = wp.empty_like(h)
    v = wp.empty_like(h)
    w = wp.empty(n, dtype=wp.vec2d)
    u = wp.empty_like(w)
    cs = wp.empty(n, dtype=wp.float64)
    ss = wp.empty_like(w)
    status = wp.zeros(1, dtype=wp.int32)
    wp.launch(schur, 1, [h, q, v, w, u, cs, ss, status, n * 100])
    assert status.numpy()[0] == 0
    ww = w.numpy()
    ww = ww[:, 0] + 1j * ww[:, 1]
    vv = v.numpy()
    vv = vv[:, :, 0] + 1j * vv[:, :, 1]
    err = np.linalg.norm(a @ vv - vv * ww, axis=0).max()
    assert err < 1.0e-9, err
    ref = np.linalg.eigvals(a)
    assert max(min(abs(ref - val)) for val in ww) < 1.0e-10
