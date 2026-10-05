import numpy as np
import pytest
import warp as wp
import warp.sparse as ws

from warpack import PartialSVD


@pytest.mark.parametrize("shape", [(50, 30), (25, 45)])
def test_rectangular_svd(shape):
    rng = np.random.default_rng(13)
    a = rng.normal(size=shape)
    r, c = np.nonzero(a)
    mat = ws.bsr_from_triplets(
        *shape,
        wp.array(r, dtype=wp.int32),
        wp.array(c, dtype=wp.int32),
        wp.array(a[r, c], dtype=wp.float64),
    )
    s = PartialSVD(mat, 5, tol=1.0e-11)
    wp.capture_launch(s.capture(100))
    sigma = s.singular_values.numpy()
    u = s.u.numpy().T
    v = s.v.numpy().T
    np.testing.assert_allclose(sigma, np.linalg.svd(a, compute_uv=False)[:5], atol=1.0e-9)
    np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1.0e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1.0e-10)
    assert np.linalg.norm(a @ v - u * sigma, axis=0).max() < 1.0e-9
    assert np.linalg.norm(a.T @ u - v * sigma, axis=0).max() < 1.0e-9
