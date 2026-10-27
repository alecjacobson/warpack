"""Paired output ownership preserves every Jacobi rotation and unique writers."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.paired_jacobi import PairedOutputJacobi, paired_update
from benchmarks.experimental.paired_jacobi_runtime import (
    RuntimePairedCompact,
    RuntimePairedJacobi,
)
from warpack import kernels as K
from warpack.projected import GlobalJacobi


@pytest.mark.parametrize("n", [2, 8, 48, 104, 208, 408, 612])
@pytest.mark.parametrize("block_dim", [64, 128, 256])
def test_round_outputs_match_existing_kernel_with_strides(n, block_dim):
    rng = np.random.default_rng(172)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    h_store = wp.full((n, n + 3), 17.0, dtype=wp.float64)
    v_store = wp.full((n, n + 3), 19.0, dtype=wp.float64)
    h, v = h_store[:, :n], v_store[:, :n]
    h.assign(a)
    v.assign(rng.normal(size=(n, n)))
    cs, ss = (wp.empty(n, dtype=wp.float64) for _ in range(2))
    partners = wp.empty(n, dtype=wp.int32)
    targets = [wp.full((n, n + 3), 23.0, dtype=wp.float64) for _ in range(4)]
    out_h, out_v, new_h, new_v = [t[:, :n] for t in targets]
    for r in sorted(set([0, 1, n // 2, n - 2])):
        if r >= n - 1:
            continue
        wp.launch(K.jacobi_pairs, n // 2, [h, cs, ss, partners, r])
        wp.launch(K.jacobi_two_sided, (n, n), [h, v, out_h, out_v, cs, ss, partners])
        wp.launch(
            paired_update, (n // 2, n // 2), [h, v, new_h, new_v, cs, ss, r], block_dim=block_dim
        )
        np.testing.assert_array_equal(new_h.numpy(), out_h.numpy())
        np.testing.assert_array_equal(new_v.numpy(), out_v.numpy())
        for target in targets:
            assert np.all(target.numpy()[:, n:] == 23.0)
    np.testing.assert_array_equal(h.numpy(), a)


def solver(constructor, n):
    h, v, temp, vtemp = [wp.empty((n, n), dtype=wp.float64) for _ in range(4)]
    cs, ss = [wp.empty(n, dtype=wp.float64) for _ in range(2)]
    return constructor(h, v, cs, ss, wp.empty(n, dtype=wp.int32), temp, vtemp)


@pytest.mark.parametrize("n", [8, 48, 104, 208, 408, 612, 1024])
@pytest.mark.parametrize("kind", ["indefinite", "repeated_null"])
def test_complete_solver_exact_control_and_independent_accuracy(n, kind):
    rng = np.random.default_rng(159)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    if kind == "repeated_null":
        q = la.qr(a)[0]
        values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
        values[: n // 3] = 0
        a = (q * values) @ q.T
    original = wp.array(a, dtype=wp.float64)
    controls = [solver(constructor, n) for constructor in [GlobalJacobi, PairedOutputJacobi]]
    graphs = []
    for eig in controls:
        wp.copy(eig.h, original)
        eig.apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            wp.copy(eig.h, original)
            eig.apply()
        graphs.append(capture.graph)
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        original.assign(a * scale)
        for graph in graphs:
            wp.capture_launch(graph)
        baseline, candidate = controls
        np.testing.assert_array_equal(candidate.h.numpy(), baseline.h.numpy())
        np.testing.assert_array_equal(candidate.v.numpy(), baseline.v.numpy())
        np.testing.assert_array_equal(candidate.sweeps.numpy(), baseline.sweeps.numpy())
        h, v = candidate.h.numpy(), candidate.v.numpy()
        if scale:
            values = np.diag(h) / scale
            assert la.norm(a @ v - v * values) / max(1, la.norm(a)) < 1e-11
            assert la.norm(v.T @ v - np.eye(n)) < 1e-11
            np.testing.assert_allclose(
                np.sort(values), la.eigvalsh(a), atol=1e-11 * max(1, la.norm(a)), rtol=1e-11
            )


@pytest.mark.parametrize(
    "constructor", [PairedOutputJacobi, RuntimePairedJacobi, RuntimePairedCompact]
)
@pytest.mark.parametrize("n", [104, 408])
def test_nonfinite_bounded_loop_and_changed_input_recovery(n, tmp_path, constructor):
    from benchmarks.graph_audit import graph_summary

    rng = np.random.default_rng(172 + n)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    source = wp.array(a, dtype=wp.float64)
    eig = solver(constructor, n)
    wp.copy(eig.h, source)
    eig.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.copy(eig.h, source)
        eig.apply()
    for kind in ["nan", "inf", "recover"]:
        matrix = a.copy()
        if kind != "recover":
            matrix[0, 0] = np.nan if kind == "nan" else np.inf
        source.assign(matrix)
        wp.capture_launch(capture.graph)
        assert 1 <= eig.sweeps.numpy()[0] <= 30
        if kind == "recover":
            h, v = eig.h.numpy(), eig.v.numpy()
            values = np.diag(h)
            assert la.norm(a @ v - v * values) / la.norm(a) < 1e-11
            assert la.norm(v.T @ v - np.eye(n)) < 1e-11
    path = tmp_path / "paired.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    audit = graph_summary(path)
    assert not audit["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
