"""Experimental search compaction keeps complete Ritz blocks and fresh checks."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.active_search import ActiveSearchBlockLOBPCG, select_search
from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator


def operator(a):
    return SparseOperator(matrix(sp.csr_matrix(a)))


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("excluded", [0, 2])
def test_captured_coupled_repeated_modes_and_changed_inputs(metric, excluded, fused, tmp_path):
    n, k, width = 64, 4, 8

    def fixture(seed):
        rng = np.random.default_rng(seed)
        lower = np.eye(n)
        if metric:
            lower += np.diag(np.linspace(0.05, 0.3 + seed * 0.001, n - 1), -1)
        mass = lower @ lower.T
        q = la.qr(rng.normal(size=(n, n)))[0]
        modes = la.solve_triangular(lower.T, q, lower=False)
        values = np.r_[1.0, 2.0, 3.0, 3.0, 4.0, 4.0, np.arange(5.0, n - 1)]
        a = mass @ (modes * values) @ modes.T @ mass
        constraints = modes[:, :excluded].T.copy()
        return a, mass, modes, values, constraints

    a, mass, modes, values, constraints = fixture(19)
    op, mass_op, pre = operator(a), operator(mass), operator(la.inv(a + mass))
    basis = wp.array(constraints, dtype=wp.float64) if excluded else None
    starts = wp.array(np.random.default_rng(27).normal(size=(k, n)), dtype=wp.float64)
    solver = ActiveSearchBlockLOBPCG(
        op,
        k,
        width,
        mass=mass_op if metric else None,
        preconditioner=pre,
        initial_vectors=starts,
        constraints=basis,
        tol=1e-10,
        symmetric_products=True,
        fused_updates=fused,
        paired_rotation=fused,
        search_bucket_size=2,
    )
    graph = solver.capture(100)
    for seed in [19, 31, 19]:
        a, mass, modes, values, constraints = fixture(seed)
        for target, host in [(op, a), (pre, la.inv(a + mass))] + (
            [(mass_op, mass)] if metric else []
        ):
            host = sp.csr_matrix(host)
            np.testing.assert_array_equal(target.matrix.columns.numpy(), host.indices)
            target.matrix.values.assign(host.data)
        if excluded:
            basis.assign(constraints)
        source = np.random.default_rng(seed + 12).normal(size=starts.shape)
        starts.assign(source)
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == k
        np.testing.assert_allclose(w, values[excluded : excluded + k], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), rtol=0, atol=1e-10)
        assert np.linalg.norm(a @ v - (mass @ v) * w, axis=0).max() < 1e-8
        wanted = modes[:, excluded : excluded + k]
        assert la.norm(v - wanted @ (wanted.T @ mass @ v)) < 1e-8
        np.testing.assert_array_equal(starts.numpy(), source)
        usage = solver.search_usage.numpy()
        assert usage.sum() == solver.iterations.numpy()[0]
        assert usage[:-1].sum() > 0
    dot = tmp_path / "active-search.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


def test_selection_reactivates_and_handles_zero_or_nonfinite_rows():
    width = 8
    errors = wp.zeros(width, dtype=wp.float64)
    rows = wp.empty(width, dtype=wp.int32)
    count = wp.zeros(1, dtype=wp.int32)
    widths = wp.array([2, 4, 6, 8], dtype=wp.int32)
    flags = wp.zeros(4, dtype=wp.int32)

    def run():
        wp.launch(select_search, 1, [errors, 1e-8, rows, count, widths, flags])

    run()
    wp.synchronize_device()
    with wp.ScopedCapture() as capture:
        run()
    for active in [[], [1, 5], list(range(8)), [3], []]:
        source = np.zeros(width)
        source[active] = 1.0
        if active:
            source[active[0]] = np.nan
        errors.assign(source)
        wp.capture_launch(capture.graph)
        assert count.numpy()[0] == len(active)
        np.testing.assert_array_equal(rows.numpy()[: len(active)], sorted(active))
        expected = np.zeros(4, dtype=np.int32)
        expected[max(0, (len(active) - 1) // 2)] = 1
        np.testing.assert_array_equal(flags.numpy(), expected)


@pytest.mark.parametrize("exact,adaptive", [(False, True), (True, True), (True, False)])
def test_odd_width_and_zero_search_fixed_or_adaptive_replay(exact, adaptive):
    n, k, width = 45, 3, 5
    a = np.diag(np.arange(1.0, n + 1))
    op = operator(a)
    start = wp.array(np.eye(n)[:width], dtype=wp.float64) if exact else None
    solver = ActiveSearchBlockLOBPCG(
        op,
        k,
        width,
        preconditioner=operator(np.diag(1 / np.diag(a))),
        initial_vectors=start,
        search_bucket_size=1,
        tol=1e-10,
    )
    graph = solver.capture(3 if exact else 100, adaptive=adaptive)
    for _ in range(2):
        wp.capture_launch(graph)
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == k
        np.testing.assert_allclose(solver.eigenvalues.numpy(), np.arange(1.0, k + 1), atol=1e-10)
        if exact:
            usage = solver.search_usage.numpy()
            assert usage[0] == (1 if adaptive else 3)
            assert usage[1:].sum() == 0
