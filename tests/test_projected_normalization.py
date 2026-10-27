"""Installed wide projected vectors must retain unit norms after long rotations."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.graph_audit import graph_summary
from tests.test_compact_jacobi import allocate
from warpack._compact_jacobi import CompactJacobi
from warpack.projected import GlobalJacobi


@pytest.mark.parametrize("constructor", [GlobalJacobi, CompactJacobi])
@pytest.mark.parametrize("n", [254, 256, 612, 1024])
def test_normalized_vectors_leave_matrix_and_native_stopping_unchanged(n, constructor, tmp_path):
    class Legacy(constructor):
        _normalize_vectors = False

    random = np.random.default_rng(159).normal(size=(n, n))
    q = la.qr((random + random.T) * 0.5)[0]
    values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
    values[: n // 3] = 0
    a = (q * values) @ q.T
    solvers, stores, graphs = [], [], []
    for kind in [Legacy, constructor]:
        eig, backing = allocate(n, constructor=kind)
        eig.h.assign(a)
        eig.apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            eig.apply()
        solvers.append(eig)
        stores.extend(backing)
        graphs.append(capture.graph)
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        for eig, graph in zip(solvers, graphs):
            eig.h.assign(a * scale)
            wp.capture_launch(graph)
        legacy, current = solvers
        np.testing.assert_array_equal(current.h.numpy(), legacy.h.numpy())
        np.testing.assert_array_equal(current.sweeps.numpy(), legacy.sweeps.numpy())
        v = current.v.numpy()
        assert la.norm(v.T @ v - np.eye(n)) < 1e-11
        if n >= 256:
            assert np.max(abs(np.sum(v * v, axis=0) - 1)) < 1e-14
        else:
            np.testing.assert_array_equal(v, legacy.v.numpy())
        if scale:
            h = current.h.numpy() / scale
            computed = np.diag(h)
            assert la.norm(a @ v - v * computed) / la.norm(a) < 1e-11
            np.testing.assert_allclose(np.sort(computed), values, atol=1e-10, rtol=1e-11)
            assert np.max(abs(h - np.diag(computed))) <= 2e-15 * np.max(abs(computed))
        for backing in stores:
            assert np.all(backing.numpy()[:, n:] == 17.0)
    path = tmp_path / "normalization.dot"
    wp.capture_debug_dot_print(graphs[1], str(path))
    audit = graph_summary(path)
    assert not audit["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
