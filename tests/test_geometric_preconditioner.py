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
from benchmarks.experimental.geometric_preconditioner import DenseBottom, build_hierarchy
from benchmarks.upstream.adapters import matrix


def dense(a):
    offsets, columns, values = a.offsets.numpy(), a.columns.numpy(), a.values.numpy()
    out = np.zeros(a.shape)
    for i in range(a.nrow):
        sl = slice(offsets[i], offsets[i + 1])
        out[i, columns[sl]] = values[sl]
    return out


def test_three_level_spd_identity_and_aliasing(tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    rng = np.random.default_rng(13403)
    n = 23
    h = rng.normal(size=(n, n))
    h = h @ h.T + np.eye(n)
    p, q = rng.normal(size=(n, 9)), rng.normal(size=(9, 4))
    a = matrix(h)
    hierarchy = build_hierarchy(a, [matrix(p), matrix(q)], n, coarse_rows=8)
    h1, h2 = p.T @ h @ p, q.T @ (p.T @ h @ p) @ q
    np.testing.assert_allclose(dense(hierarchy.coarse.matrix), h1, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(dense(hierarchy.coarse.coarse.matrix), h2, rtol=2e-13, atol=2e-11)
    g0 = dense(hierarchy.smoother.factor.G)
    g1 = dense(hierarchy.coarse.smoother.factor.G)
    expected = g0.T @ g0 + p @ (g1.T @ g1 + q @ np.linalg.solve(h2, q.T)) @ p.T
    assert np.linalg.eigvalsh(expected).min() > 0
    x = wp.array(np.eye(n), dtype=wp.float64)
    hierarchy.apply(x, x)
    with wp.ScopedCapture() as capture:
        hierarchy.apply(x, x)
    for rhs in (np.eye(n), rng.normal(size=(n, n)), np.eye(n)):
        x.assign(rhs)
        wp.capture_launch(capture.graph)
        assert hierarchy.status.numpy()[0] == 0
        np.testing.assert_allclose(x.numpy(), rhs @ expected, atol=5e-13, rtol=1e-11)
    audit(capture.graph, tmp_path / "hierarchy.dot")


def test_bottom_refactor_rejects_indefinite_and_recovers(tmp_path):
    rng = np.random.default_rng(64)
    h = rng.normal(size=(11, 11))
    h = h @ h.T + np.eye(11)
    a = matrix(h)
    bottom = DenseBottom(a, 3)
    rhs = rng.normal(size=(3, 11))
    x = wp.array(rhs, dtype=wp.float64)
    bottom.apply(x, x)
    with wp.ScopedCapture() as capture:
        bottom.update()
        bottom.apply(x, x)
    for scale in (1.0, -1.0, 0.7):
        a.values.assign((scale * h).ravel())
        x.assign(rhs)
        wp.capture_launch(capture.graph)
        if scale < 0:
            assert bottom.status.numpy()[0] != 0
            assert np.isnan(x.numpy()).all()
        else:
            assert bottom.status.numpy()[0] == 0
            np.testing.assert_allclose(x.numpy(), np.linalg.solve(scale * h, rhs.T).T, atol=1e-13)
    audit(capture.graph, tmp_path / "bottom.dot")


@pytest.mark.parametrize("consistent", [False, True])
def test_warp_galerkin_metric_and_generalized_fem(consistent, tmp_path):
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
        hierarchy = build_hierarchy(shifted, [transfer], width, fine_factor=factor)
        return CoarseCG(shifted, hierarchy, width)

    fine = ShapeReplay([(0, p, 0)], 8, refit=False, preconditioner_factory=factory)
    expected = (host_transfer.T @ (p.a + 0.01 * p.b) @ host_transfer).toarray()
    np.testing.assert_allclose(dense(hierarchy.coarse.matrix), expected, atol=2e-13, rtol=1e-11)
    for _ in range(2):
        fine.solve()
        q = fine.quality(p)
        assert q["accepted"], q
        assert hierarchy.status.numpy()[0] == 0
    audit(fine.graph, tmp_path / "fem.dot")
