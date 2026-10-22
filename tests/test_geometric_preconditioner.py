"""Independent additive SPD identity, Galerkin construction and captured FEM solve."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import (
    attach_reference,
    audit,
    beam_problem,
    transfer_stencil,
)
from benchmarks.applications.fem_shapes import ShapeReplay
from benchmarks.experimental.coarse_correction import CoarseCG
from benchmarks.experimental.geometric_cg import GeometricCG
from benchmarks.experimental.geometric_preconditioner import DenseBottom, build_hierarchy
from benchmarks.upstream.adapters import matrix
from warpack._active_cg import DeviceBucketedCG


def dense(a):
    offsets, columns, values = a.offsets.numpy(), a.columns.numpy(), a.values.numpy()
    out = np.zeros(a.shape)
    for i in range(a.nrow):
        sl = slice(offsets[i], offsets[i + 1])
        out[i, columns[sl]] = values[sl]
    return out


@pytest.mark.parametrize("dtype", [wp.float64, wp.float32])
@pytest.mark.parametrize("columns", [False, True])
def test_three_level_spd_identity_and_aliasing(tmp_path, dtype, columns):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    rng = np.random.default_rng(13403)
    n = 23
    h = rng.normal(size=(n, n))
    h = h @ h.T + np.eye(n)
    p, q = rng.normal(size=(n, 9)), rng.normal(size=(9, 4))
    a = matrix(h)
    hierarchy = build_hierarchy(
        a, [matrix(p), matrix(q)], n, coarse_rows=8, dtype=dtype, columns=columns
    )
    h1, h2 = p.T @ h @ p, q.T @ (p.T @ h @ p) @ q
    np.testing.assert_allclose(dense(hierarchy.coarse.matrix), h1, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(dense(hierarchy.coarse.coarse.matrix), h2, rtol=2e-13, atol=2e-11)
    g0 = dense(hierarchy.smoother.factor.G)
    g1 = dense(hierarchy.coarse.smoother.factor.G)
    expected = g0.T @ g0 + p @ (g1.T @ g1 + q @ np.linalg.solve(h2, q.T)) @ p.T
    assert np.linalg.eigvalsh(expected).min() > 0
    x = wp.array(np.eye(n), dtype=dtype)
    hierarchy.apply(x, x)
    with wp.ScopedCapture() as capture:
        hierarchy.apply(x, x)
    for rhs in (np.eye(n), rng.normal(size=(n, n)), np.eye(n)):
        x.assign(rhs)
        wp.capture_launch(capture.graph)
        assert hierarchy.status.numpy()[0] == 0
        np.testing.assert_allclose(
            x.numpy(),
            rhs @ expected,
            atol=5e-13 if dtype == wp.float64 else 2e-7,
            rtol=1e-11 if dtype == wp.float64 else 1e-4,
        )
    audit(capture.graph, tmp_path / "hierarchy.dot")


@pytest.mark.parametrize("columns", [False, True])
def test_bottom_refactor_rejects_indefinite_and_recovers(tmp_path, columns):
    rng = np.random.default_rng(64)
    h = rng.normal(size=(11, 11))
    h = h @ h.T + np.eye(11)
    a = matrix(h)
    bottom = DenseBottom(a, 3)
    rhs = rng.normal(size=(3, 11))
    x = wp.array(rhs.T.copy() if columns else rhs, dtype=wp.float64)
    apply = bottom.apply_columns if columns else bottom.apply
    apply(x, x)
    with wp.ScopedCapture() as capture:
        bottom.update()
        apply(x, x)
    for scale in (1.0, -1.0, 0.7):
        a.values.assign((scale * h).ravel())
        x.assign(rhs.T.copy() if columns else rhs)
        wp.capture_launch(capture.graph)
        if scale < 0:
            assert bottom.status.numpy()[0] != 0
            assert np.isnan(x.numpy()).all()
        else:
            assert bottom.status.numpy()[0] == 0
            result = x.numpy().T if columns else x.numpy()
            np.testing.assert_allclose(result, np.linalg.solve(scale * h, rhs.T).T, atol=1e-13)
    audit(capture.graph, tmp_path / "bottom.dot")


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("bucketed,columns", [(False, False), (True, False), (True, True)])
def test_warp_galerkin_metric_and_generalized_fem(consistent, bucketed, columns, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(2, 6, consistent)
    attach_reference(p)
    ids, weights = transfer_stencil(1, 2)
    ids, weights = ids.ravel(), weights.ravel()
    valid = (ids >= 0) & (weights != 0)
    host_transfer = sp.csr_matrix(
        (weights[valid], (np.repeat(np.arange(p.n), 4)[valid], ids[valid])), shape=(p.n, 36)
    )
    transfer = matrix(host_transfer)
    hierarchy = None

    def factory(shifted, factor, width):
        nonlocal hierarchy
        hierarchy = build_hierarchy(
            shifted,
            [transfer],
            width,
            fine_factor=factor,
            dtype=wp.float32 if bucketed else wp.float64,
            columns=columns,
        )
        if bucketed:
            return GeometricCG(shifted, hierarchy)
        return CoarseCG(shifted, hierarchy, width)

    fine = ShapeReplay(
        [(0, p, 0)],
        8,
        refit=False,
        preconditioner_factory=factory,
        preconditioner_bucket_size=4 if bucketed else 0,
    )
    expected = (host_transfer.T @ (p.a + 0.01 * p.b) @ host_transfer).toarray()
    np.testing.assert_allclose(dense(hierarchy.coarse.matrix), expected, atol=2e-13, rtol=1e-11)
    for _ in range(2):
        fine.solve()
        q = fine.quality(p)
        assert q["accepted"], q
        assert hierarchy.status.numpy()[0] == 0
        if bucketed:
            assert np.count_nonzero(fine.solver.preconditioner.usage.numpy()[:-1]) > 0
            assert fine.cg_statistics()[3] > 0
    audit(fine.graph, tmp_path / "fem.dot")


@pytest.mark.parametrize("columns", [False, True])
def test_shared_width_cg_changed_matrix_zero_scaled_and_alias_rhs(tmp_path, columns):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    rng = np.random.default_rng(13432)
    n = 23
    a = rng.normal(size=(n, n))
    a = a @ a.T + 3 * np.eye(n)
    source = matrix(a)
    hierarchy = build_hierarchy(
        source, [matrix(rng.normal(size=(n, 4)))], 7, dtype=wp.float32, columns=columns
    )
    cg = GeometricCG(source, hierarchy, tol=1e-5, maxiter=60)
    states = [cg, cg.for_width(3), cg.for_width(1)]
    assert all(s.operator.matrix is cg.operator.matrix for s in states)
    assert all(s.hierarchy.smoother.factor is hierarchy.smoother.factor for s in states)
    assert all(s.hierarchy.coarse.lower is hierarchy.coarse.lower for s in states)
    assert len({s.hierarchy.rhs.ptr for s in states}) == 3
    rows = [wp.empty((s.width, n), dtype=wp.float64) for s in states]
    for s, x in zip(states, rows):
        x.zero_()
        s.apply(x, x)
    with wp.ScopedCapture() as capture:
        for s, x in zip(states, rows):
            s.reset_stats()
            s.apply(x, x)
    for scale, rhs_scale in ((1.0, 1.0), (1.2, 0.0), (0.7, 1e-8), (1.0, 1e3)):
        source.values.assign((a * scale).ravel())
        rhs = [rng.normal(size=x.shape) * rhs_scale for x in rows]
        for x, b in zip(rows, rhs):
            x.assign(b)
        wp.capture_launch(capture.graph)
        for s, x, b in zip(states, rows, rhs):
            residual = x.numpy() @ (a * scale) - b
            if rhs_scale == 0:
                np.testing.assert_array_equal(x.numpy(), np.zeros_like(b))
            else:
                relative = np.linalg.norm(residual, axis=1) / np.linalg.norm(b, axis=1)
                assert max(relative) < 3e-5
            assert s.stats.numpy()[3] == s.width
    audit(capture.graph, tmp_path / "mixed-cg.dot")

    errors = wp.zeros(7, dtype=wp.float64)
    bucket = DeviceBucketedCG(cg, errors, 0.1, bucket_size=2)
    x = rows[0]
    with wp.ScopedCapture() as capture:
        bucket.reset_stats()
        bucket.apply(x, x)
    for count in (7, 1, 3, 0, 7):
        active = np.arange(7) >= 7 - count  # Exercise packing non-leading active rows.
        errors.assign(active.astype(float))
        rhs = rng.normal(size=x.shape)
        x.assign(rhs)
        wp.capture_launch(capture.graph)
        result = x.numpy()
        np.testing.assert_array_equal(result[~active], np.zeros_like(result[~active]))
        if count:
            residual = result[active] @ a - rhs[active]
            assert (
                np.max(np.linalg.norm(residual, axis=1) / np.linalg.norm(rhs[active], axis=1))
                < 3e-5
            )
        usage = bucket.usage.numpy()
        assert usage[-1] == count
        assert np.sum(usage[:-1]) == int(count > 0)
        assert bucket.stats.numpy()[3] == (0 if count == 0 else min(7, 2 * ((count + 1) // 2)))
    audit(capture.graph, tmp_path / "mixed-buckets.dot")
