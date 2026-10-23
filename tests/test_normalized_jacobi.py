"""Column normalization must repair norms without hiding vector correlations."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.normalized_jacobi import LegacyJacobi, NormalizedJacobi


def create(constructor, n):
    h, v, tmp, vt = [wp.empty((n, n), dtype=wp.float64) for _ in range(4)]
    cs, ss = [wp.empty(n, dtype=wp.float64) for _ in range(2)]
    return constructor(h, v, cs, ss, wp.empty(n, dtype=wp.int32), tmp, vt)


@pytest.mark.parametrize("n", [104, 612, 1024])
@pytest.mark.parametrize("kind", ["indefinite", "repeated_null"])
def test_column_norm_recovery_and_independent_full_basis(n, kind, tmp_path):
    from benchmarks.audit_profile import graph_summary

    rng = np.random.default_rng(159)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    if kind == "repeated_null":
        q = la.qr(a)[0]
        values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
        values[: n // 3] = 0
        a = (q * values) @ q.T
    values = la.eigvalsh(a)
    source = wp.array(a, dtype=wp.float64)
    solvers = [create(c, n) for c in [LegacyJacobi, NormalizedJacobi]]
    graphs = []
    for eig in solvers:
        wp.copy(eig.h, source)
        eig.apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            wp.copy(eig.h, source)
            eig.apply()
        graphs.append(capture.graph)
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        source.assign(a * scale)
        for graph in graphs:
            wp.capture_launch(graph)
        raw, fixed = solvers
        np.testing.assert_array_equal(raw.h.numpy(), fixed.h.numpy())
        np.testing.assert_array_equal(raw.sweeps.numpy(), fixed.sweeps.numpy())
        v = fixed.v.numpy()
        assert la.norm(v.T @ v - np.eye(n)) < 1e-11
        assert np.max(abs(np.sum(v * v, axis=0) - 1)) < 1e-14
        if scale:
            computed = np.diag(fixed.h.numpy()) / scale
            assert la.norm(a @ v - v * computed) / max(1, la.norm(a)) < 1e-11
            np.testing.assert_allclose(
                np.sort(computed), values, atol=1e-11 * max(1, la.norm(a)), rtol=1e-11
            )
    path = tmp_path / "normalized.dot"
    wp.capture_debug_dot_print(graphs[1], str(path))
    audit = graph_summary(path)
    assert not audit["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


@pytest.mark.parametrize("n", [104, 512])
def test_normalize_column_invalid_preservation_strides_and_recovery(n):
    from benchmarks.experimental.normalized_jacobi import normalize_columns

    rng = np.random.default_rng(174)
    initial = rng.normal(size=(n, 8))
    backing = wp.full((n, 11), 23.0, dtype=wp.float64)
    v = backing[:, :8]
    v.assign(initial)
    wp.launch_tiled(normalize_columns, dim=8, inputs=[v], block_dim=128)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.launch_tiled(normalize_columns, dim=8, inputs=[v], block_dim=128)
    bad = initial.copy()
    bad[0, 0] = np.nan
    bad[:, 1] = 0
    bad[0, 2] = np.inf
    for original in [bad, initial]:
        v.assign(original)
        wp.capture_launch(capture.graph)
        out = v.numpy()
        for j in range(8):
            column = original[:, j]
            if not np.all(np.isfinite(column)) or not np.any(column):
                np.testing.assert_array_equal(out[:, j], column)
            else:
                extended = out[:, j].astype(np.longdouble)
                assert abs(np.sum(extended * extended) - 1) < 1e-14
                np.testing.assert_allclose(
                    out[:, j], column / la.norm(column), atol=1e-15, rtol=1e-14
                )
        assert np.all(backing.numpy()[:, 8:] == 23.0)
