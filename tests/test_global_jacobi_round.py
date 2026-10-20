"""Fused global rounds match independent dense plane rotations and replay."""

import numpy as np
import pytest
import warp as wp

from warpack import kernels as K


@pytest.mark.parametrize("n", [8, 24, 48, 64, 72, 96, 128])
@pytest.mark.parametrize("last_round", [False, True])
def test_global_jacobi_fused_round(n, last_round, tmp_path):
    rng = np.random.default_rng(837)
    a = rng.normal(size=(n, n))
    a += a.T
    v = np.linalg.qr(rng.normal(size=(n, n)))[0]
    source_h = wp.array(a, dtype=wp.float64)
    source_v = wp.array(v, dtype=wp.float64)
    old_h, old_v, new_h, new_v, temp, vtemp = [wp.empty_like(source_h) for _ in range(6)]
    c, s = [wp.empty(n, dtype=wp.float64) for _ in range(2)]
    partners = wp.empty(n, dtype=wp.int32)
    rotation = n - 2 if last_round else 0

    def run():
        wp.copy(old_h, source_h)
        wp.copy(old_v, source_v)
        wp.launch(K.jacobi_pairs, n // 2, [source_h, c, s, partners, rotation])
        wp.launch(K.jacobi_right, (n, n), [old_h, old_v, temp, vtemp, c, s, partners])
        wp.launch(K.jacobi_left, (n, n), [old_h, old_v, temp, vtemp, c, s, partners])
        wp.launch(K.jacobi_two_sided, (n, n), [source_h, source_v, new_h, new_v, c, s, partners])

    run()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        run()
    for factor in [1e-50, -0.37, 1e6]:
        original = a * factor
        source_h.assign(original)
        wp.capture_launch(capture.graph)
        cc, ss, pp = c.numpy(), s.numpy(), partners.numpy()
        rotation_matrix = np.diag(cc)
        rotation_matrix[pp, np.arange(n)] = -ss
        expected_h = rotation_matrix.T @ original @ rotation_matrix
        expected_v = v @ rotation_matrix
        actual_h, actual_v = new_h.numpy(), new_v.numpy()
        scale = np.linalg.norm(original)
        assert np.linalg.norm(actual_h - expected_h) / scale < 1e-13
        assert np.linalg.norm(actual_h - old_h.numpy()) / scale < 1e-13
        assert np.linalg.norm(actual_h - actual_h.T) / scale < 1e-13
        np.testing.assert_allclose(actual_v, expected_v, rtol=0, atol=1e-13)
        np.testing.assert_allclose(actual_v, old_v.numpy(), rtol=0, atol=1e-13)
        np.testing.assert_allclose(actual_v.T @ actual_v, np.eye(n), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(source_h.numpy(), original)
        np.testing.assert_array_equal(source_v.numpy(), v)
    dot = tmp_path / "fused-round.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    assert not any(word in dot.read_text() for word in ("HOST", "HtoD", "DtoH"))
