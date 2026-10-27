"""Immutable partner-derived rounds preserve compact-loop state and arithmetic."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.paired_jacobi_runtime import (
    RuntimePairedCompact,
    RuntimePairedJacobi,
    update,
)
from benchmarks.graph_audit import graph_summary
from tests.test_compact_jacobi import allocate
from warpack import kernels as K
from warpack._compact_jacobi import CompactJacobi
from warpack.projected import GlobalJacobi


@pytest.mark.parametrize("n", [2, 8, 104, 256, 312, 408, 612, 1024])
def test_immutable_round_and_exact_outputs(n):
    rng = np.random.default_rng(172)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    source = wp.array(a, dtype=wp.float64)
    v = wp.array(rng.normal(size=(n, n)), dtype=wp.float64)
    cs, ss = [wp.empty(n, dtype=wp.float64) for _ in range(2)]
    partners = wp.empty(n, dtype=wp.int32)
    targets = [wp.full((n, n + 3), 17.0, dtype=wp.float64) for _ in range(4)]
    old_h, old_v, new_h, new_v = [a[:, :n] for a in targets]
    for r in sorted(set([0, 1, n // 2, n - 2])):
        if r >= n - 1:
            continue
        wp.launch(K.jacobi_pairs, n // 2, [source, cs, ss, partners, r])
        assert partners.numpy()[-1] == r
        wp.launch(K.jacobi_two_sided, (n, n), [source, v, old_h, old_v, cs, ss, partners])
        wp.launch(
            update, (n // 2, n // 2), [source, v, new_h, new_v, cs, ss, partners], block_dim=64
        )
        np.testing.assert_array_equal(new_h.numpy(), old_h.numpy())
        np.testing.assert_array_equal(new_v.numpy(), old_v.numpy())
    for a in targets:
        assert np.all(a.numpy()[:, n:] == 17.0)


@pytest.mark.parametrize("n", [8, 254, 256, 408, 612, 1024])
@pytest.mark.parametrize("rounds", [2, 16])
def test_global_and_compact_exact_full_solve_replay(n, rounds, tmp_path):
    rng = np.random.default_rng(159)
    a = rng.normal(size=(n, n))
    a = (a + a.T) * 0.5
    q = la.qr(a)[0]
    values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
    values[: n // 3] = 0
    a = (q * values) @ q.T
    constructors = [
        GlobalJacobi,
        RuntimePairedJacobi,
        partial(CompactJacobi, rounds_per_loop=rounds),
        partial(RuntimePairedCompact, rounds_per_loop=rounds),
    ]
    solvers, graphs, backings = [], [], []
    for constructor in constructors:
        eig, stores = allocate(n, constructor=constructor)
        eig.h.assign(a)
        eig.apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            eig.apply()
        solvers.append(eig)
        graphs.append(capture.graph)
        backings.extend(stores)
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        outputs = []
        for eig, graph in zip(solvers, graphs):
            eig.h.assign(a * scale)
            wp.capture_launch(graph)
            outputs.append((eig.h.numpy(), eig.v.numpy(), eig.sweeps.numpy()))
            if isinstance(eig, CompactJacobi):
                assert eig.round_index.numpy()[0] == eig.loop_rounds
        for candidate in outputs[1:]:
            for actual, expected in zip(candidate, outputs[0]):
                np.testing.assert_array_equal(actual, expected)
        h, v, _ = outputs[0]
        assert la.norm(v.T @ v - np.eye(n)) < 1e-11
        if scale:
            computed = np.diag(h) / scale
            assert la.norm(a @ v - v * computed) / max(1, la.norm(a)) < 1e-11
            np.testing.assert_allclose(np.sort(computed), values, atol=1e-10, rtol=1e-11)
        for store in backings:
            assert np.all(store.numpy()[:, n:] == 17.0)
    path = tmp_path / "paired-compact.dot"
    wp.capture_debug_dot_print(graphs[-1], str(path))
    audit = graph_summary(path)
    assert not audit["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
