"""Logical CSR coefficients include every stored duplicate, on every replay."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws

from warpack import NormalSVD, PartialSVD
from warpack.constrained import _constraint_norm_parts, _constraint_norms
from warpack.sparse_utils import _Transpose


def duplicate_matrix(host, *, device=None, unsorted=True):
    """Raw compact CSR: three entries per coefficient, including cancellation."""
    host = np.asarray(host)
    cols = np.tile(np.repeat(np.arange(host.shape[1]), 3), (host.shape[0], 1))
    values = np.stack((0.25 * host, 0.75 * host + 0.125, np.full_like(host, -0.125)), axis=2)
    values = values.reshape(host.shape[0], -1)
    if unsorted:
        order = np.random.default_rng(48).permutation(cols.shape[1])
        cols, values = cols[:, order], values[:, order]
    offsets = np.arange(host.shape[0] + 1) * cols.shape[1]
    a = ws.bsr_zeros(*host.shape, wp.float64, device=device)
    a.offsets = wp.array(np.r_[offsets, [999999]], dtype=wp.int32, device=device)
    a.columns = wp.array(np.r_[cols.ravel(), [-99] * 7], dtype=wp.int32, device=device)
    a.values = wp.array(np.r_[values.ravel(), [np.nan] * 7], dtype=wp.float64, device=device)
    a.nnz = int(offsets[-1])
    return a


def host_matrix(a):
    offsets = a.offsets.numpy()[: a.nrow + 1]
    count = offsets[-1]
    return sp.csr_matrix(
        (a.values.numpy()[:count], a.columns.numpy()[:count], offsets), shape=a.shape
    ).toarray()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("unsorted", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_duplicate_transpose_and_row_norms(device, unsorted, wide, tmp_path):
    h = np.random.default_rng(127).normal(size=(7, 11) if wide else (11, 7))
    h[2] = 0
    a = duplicate_matrix(h, device=device, unsorted=unsorted)
    cache = _Transpose(a, with_source_owners=True)
    parts = wp.empty((a.nrow, 1), dtype=wp.float64, device=device)
    norms = wp.empty(a.nrow, dtype=wp.float64, device=device)

    def apply():
        cache.refresh()
        if device == "cpu":
            return  # Cooperative constraint reductions are CUDA operations.
        wp.launch_tiled(
            _constraint_norm_parts,
            dim=parts.shape,
            inputs=[a.offsets, cache.indices, cache.owners, cache.matrix.values, parts],
            block_dim=128,
            device=device,
        )
        wp.launch(_constraint_norms, a.nrow, [parts, norms], device=device)

    apply()
    if device != "cpu":
        wp.synchronize()
        with wp.ScopedCapture(device=device) as cap:
            apply()
        dot = tmp_path / "duplicate-refresh.dot"
        wp.capture_debug_dot_print(cap.graph, str(dot))
        assert not any(word in dot.read_text() for word in ["HOST", "DtoH", "HtoD"])
    original = a.values.numpy()
    changed = original.copy()
    changed[: a.nnz] += np.linspace(-0.3, 0.7, a.nnz)
    for values in [original, 2 * original, changed, np.zeros_like(original), original]:
        a.values.assign(values)
        if device == "cpu":
            apply()
        else:
            wp.capture_launch(cap.graph)
        expected = host_matrix(a)
        np.testing.assert_allclose(host_matrix(cache.matrix), expected.T, atol=3e-15, rtol=1e-14)
        if device != "cpu":
            np.testing.assert_allclose(
                norms.numpy(), la.norm(expected, axis=1), atol=3e-15, rtol=1e-14
            )
        np.testing.assert_array_equal(a.values.numpy(), values)


@pytest.mark.parametrize("solver_type", [NormalSVD, PartialSVD])
@pytest.mark.parametrize("wide", [False, True])
def test_duplicate_svd_replay(solver_type, wide, tmp_path):
    h = np.random.default_rng(949).normal(size=(61, 23))
    h[3] = 0
    if wide:
        h = h.T
    a = duplicate_matrix(h)
    ncv = 23 if solver_type is NormalSVD else 32
    solver = solver_type(a, 5, ncv=ncv, tol=1e-11)
    graph = solver.capture(100)
    original = a.values.numpy()
    changed = original.copy()
    changed[: a.nnz] *= np.random.default_rng(414).uniform(0.7, 1.3, a.nnz)
    for values in [original, 2 * original, changed, original]:
        a.values.assign(values)
        wp.capture_launch(graph)
        host = host_matrix(a)
        u0, w0, vh0 = la.svd(host, full_matrices=False)
        w, u, v = solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T
        np.testing.assert_allclose(w, w0[:5], atol=1e-9, rtol=0)
        np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1e-10)
        np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1e-10)
        assert la.norm(host @ v - u * w, axis=0).max() < 1e-9
        assert la.norm(host.T @ u - v * w, axis=0).max() < 1e-9
        assert la.norm(u - u0[:, :5] @ (u0[:, :5].T @ u)) < 1e-9
        assert la.norm(v - vh0[:5].T @ (vh0[:5] @ v)) < 1e-9
        assert solver.core.status.numpy()[0] == 0
        assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 5
        np.testing.assert_array_equal(a.values.numpy(), values)
    dot = tmp_path / "duplicate-svd.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(word in dot.read_text() for word in ["HOST", "DtoH", "HtoD"])
