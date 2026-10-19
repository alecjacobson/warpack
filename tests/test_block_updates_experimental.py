"""Independent products for optional experimental FP64 outer updates."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.block_updates import (
    paired_ritz_rotation,
    subtract_projection,
    symmetric_gram_kernel,
)


@pytest.mark.parametrize("chunk", [256, 4096])
@pytest.mark.parametrize("width", [6, 13, 24])
def test_symmetric_product_partial_tiles_and_replay(chunk, width):
    rng = np.random.default_rng(387)
    n = 4171
    original = rng.normal(size=(width, n))
    # Indefinite diagonal A; symmetry does not rely on positive definiteness.
    diagonal = rng.normal(size=n)
    x = wp.array(original, dtype=wp.float64)
    y = wp.array(original * diagonal, dtype=wp.float64)
    g = wp.empty((width, width), dtype=wp.float64)

    def apply():
        g.zero_()
        wp.launch_tiled(
            symmetric_gram_kernel(chunk),
            dim=((width + 7) // 8, (width + 7) // 8, (n + chunk - 1) // chunk),
            inputs=[x, y, g],
            block_dim=128,
        )

    apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        apply()
    for scale in (1.0, -1e-12, 1e12):
        x.assign(original * scale)
        y.assign(original * diagonal * scale)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(
            g.numpy() / scale**2, original @ (original * diagonal).T, atol=3e-11, rtol=3e-12
        )


@pytest.mark.parametrize("width,basis", [(6, 3), (13, 9), (24, 48)])
def test_fused_projection_partial_tiles_and_changed_input(width, basis):
    rng = np.random.default_rng(47)
    n = 137
    qh, ch, xh = (
        rng.normal(size=(basis, n)),
        rng.normal(size=(width, basis)),
        rng.normal(size=(width, n)),
    )
    q, c, x = [wp.array(h, dtype=wp.float64) for h in (qh, ch, xh)]

    def apply():
        wp.launch_tiled(
            subtract_projection,
            dim=((width + 3) // 4, (n + 31) // 32),
            inputs=[q, c, x],
            block_dim=128,
        )

    apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        apply()
    for scale in (1.0, -1e-12, 1e12):
        q.assign(qh * scale)
        x.assign(xh * scale)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(x.numpy() / scale, xh - ch @ qh, atol=5e-13, rtol=2e-12)


@pytest.mark.parametrize("width", [6, 13, 24])
def test_paired_ritz_rotations_partial_inner_tiles_and_replay(width):
    rng = np.random.default_rng(412)
    n = 137
    sh, ch = rng.normal(size=(3 * width, n)), rng.normal(size=(width, 3 * width))
    s, c = wp.array(sh, dtype=wp.float64), wp.array(ch, dtype=wp.float64)
    x, p = [wp.empty((width, n), dtype=wp.float64) for _ in range(2)]

    def apply():
        wp.launch_tiled(
            paired_ritz_rotation,
            dim=((width + 3) // 4, (n + 31) // 32),
            inputs=[s[:width], s[width:], c[:, :width], c[:, width:], x, p],
            block_dim=128,
        )

    apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        apply()
    for scale in (1.0, -1e-12, 1e12):
        s.assign(sh * scale)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(x.numpy() / scale, ch @ sh, atol=5e-13, rtol=2e-12)
        np.testing.assert_allclose(
            p.numpy() / scale, ch[:, width:] @ sh[width:], atol=5e-13, rtol=2e-12
        )
