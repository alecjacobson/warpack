import numpy as np
import pytest
import warp as wp

from warpack.projected import jacobi_kernel


@pytest.mark.parametrize("block_dim", [256, 512])
@pytest.mark.parametrize("n", [10, 24, 32, 48, 64])
@pytest.mark.parametrize("kind", ["random", "clustered", "zero", "scaled"])
def test_projected_eigenproblem(n, kind, block_dim):
    rng = np.random.default_rng(72)
    a = rng.normal(size=(n, n))
    a += a.T
    if kind == "clustered":
        q, _ = np.linalg.qr(a)
        a = (q * (1 + np.arange(n) * 1e-9)) @ q.T
    if kind == "zero":
        a.fill(0)
    if kind == "scaled":
        a *= 1e-50
    scale = max(np.linalg.norm(a), 1e-100)
    da = wp.array(a, dtype=wp.float64)
    dv = wp.empty_like(da)
    kernel = jacobi_kernel(n, block_dim)
    wp.launch_tiled(kernel, dim=1, inputs=[da, dv], block_dim=block_dim)
    source = wp.array(a, dtype=wp.float64)
    with wp.ScopedCapture() as cap:
        wp.copy(da, source)
        wp.launch_tiled(kernel, dim=1, inputs=[da, dv], block_dim=block_dim)
    wp.capture_launch(cap.graph)
    v = dv.numpy()
    values = np.diag(da.numpy())
    assert np.linalg.norm(a @ v - v * values) / scale < 1e-13
    assert np.linalg.norm(v.T @ v - np.eye(n)) < 1e-12
    np.testing.assert_allclose(np.sort(values) / scale, np.linalg.eigvalsh(a) / scale, atol=1e-13)
