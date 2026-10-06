import numpy as np
import warp as wp

from benchmarks.mesh_quality import tet_metrics


def test_regular_tetrahedron_metrics_ignore_vertex_orientation():
    x = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]]
    )
    t = np.array([[0, 1, 2, 3], [0, 2, 1, 3]], dtype=np.int32)
    result = wp.empty((2, 5), dtype=wp.float64)
    wp.launch(tet_metrics, 2, [wp.array(x, dtype=wp.vec3d), wp.array(t, dtype=wp.vec4i), result])
    values = result.numpy()
    np.testing.assert_allclose(values[:, 0], 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(values[:, 1], np.sqrt(2) / 12, atol=1e-14, rtol=0)
    np.testing.assert_allclose(values[:, 2], np.degrees(np.arccos(1 / 3)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(values[:, 3], 1, atol=1e-14, rtol=0)
    assert values[0, 4] > 0 and values[1, 4] < 0
