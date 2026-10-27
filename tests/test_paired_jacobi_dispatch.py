"""Installed wide update must retain legacy arithmetic and compact loop state."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.graph_audit import graph_summary
from tests.test_compact_jacobi import allocate
from warpack import kernels as K
from warpack._compact_jacobi import CompactJacobi
from warpack.projected import GlobalJacobi


class LegacyGlobal(GlobalJacobi):
    _paired_outputs = False


class LegacyCompact(CompactJacobi):
    _paired_outputs = False


@pytest.mark.parametrize("n", [254, 256, 408, 612, 1024])
@pytest.mark.parametrize("compact", [False, True])
def test_installed_dispatch_matches_legacy_and_original_spectrum(n, compact, tmp_path):
    rng = np.random.default_rng(159)
    a = rng.normal(size=(n, n))
    q = la.qr((a + a.T) * 0.5)[0]
    values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
    values[: n // 3] = 0
    a = (q * values) @ q.T
    constructors = [LegacyGlobal, GlobalJacobi]
    if compact:
        constructors = [partial(c, rounds_per_loop=16) for c in [LegacyCompact, CompactJacobi]]
    solvers, graphs, stores = [], [], []
    for construct in constructors:
        solver, backing = allocate(n, constructor=construct)
        solver.h.assign(a)
        solver.apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            solver.apply()
        solvers.append(solver)
        graphs.append(capture.graph)
        stores.extend(backing)
    assert not solvers[0]._paired_update
    expected = wp.get_device().arch == 89 and n >= 256
    assert solvers[1]._paired_update == expected
    assert solvers[1]._update_kernel is (
        K.jacobi_two_sided_paired if expected else K.jacobi_two_sided
    )
    for scale in [1.0, 1e-80, 1e80, 0.0, 1.0]:
        outputs = []
        for solver, graph in zip(solvers, graphs):
            solver.h.assign(a * scale)
            wp.capture_launch(graph)
            outputs.append((solver.h.numpy(), solver.v.numpy(), solver.sweeps.numpy()))
            if compact:
                assert solver.round_index.numpy()[0] == solver.loop_rounds
        for installed, legacy in zip(outputs[1], outputs[0]):
            np.testing.assert_array_equal(installed, legacy)
        h, v, sweeps = outputs[1]
        assert 1 <= sweeps[0] <= 30
        assert la.norm(v.T @ v - np.eye(n)) < 1e-11
        if scale:
            eigenvalues = np.diag(h) / scale
            np.testing.assert_allclose(np.sort(eigenvalues), values, atol=1e-10, rtol=1e-11)
            assert la.norm(a @ v - v * eigenvalues) / max(1, la.norm(a)) < 1e-11
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)
    path = tmp_path / "installed.dot"
    wp.capture_debug_dot_print(graphs[1], str(path))
    audit = graph_summary(path)
    assert not audit["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
