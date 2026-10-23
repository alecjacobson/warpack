"""Partial tiles and strides in structured inverse application."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.lower_triangular_rotate import lower_rotate
from warpack.block import rotate


@pytest.mark.parametrize("width", [3, 7, 51, 104, 204])
def test_skipped_zero_tiles_preserve_dense_product_and_sentinels(width):
    n = 113
    rng = np.random.default_rng(173)
    q = rng.normal(size=(width, n))
    coeff = np.tril(rng.normal(size=(width, width)))
    qstore = wp.full((width, n + 5), 17.0, dtype=wp.float64)
    cstore = wp.full((width, width + 3), 19.0, dtype=wp.float64)
    source = qstore[:, :n]
    lower = cstore[:, :width]
    lower.assign(coeff)
    stores = [wp.full((width, n + 5), 23.0, dtype=wp.float64) for _ in range(2)]
    out = [s[:, :n] for s in stores]
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        source.assign(q * scale)
        rotate(source, lower, out[0])
        wp.launch_tiled(
            lower_rotate,
            dim=((width + 3) // 4, (n + 31) // 32),
            inputs=[source, lower, out[1]],
            block_dim=128,
        )
        np.testing.assert_array_equal(out[0].numpy(), out[1].numpy())
        np.testing.assert_allclose(
            out[1].numpy(), coeff @ (q * scale), atol=1e-12 * abs(scale), rtol=1e-11
        )
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 23.0)
        assert np.all(qstore.numpy()[:, n:] == 17.0)
        assert np.all(cstore.numpy()[:, width:] == 19.0)
