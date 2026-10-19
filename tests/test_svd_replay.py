"""Changed fixed-topology SVD inputs must refresh the cached transpose on device."""

import re

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws

from benchmarks.experimental.normal_svd import NormalSVD
from benchmarks.upstream.adapters import matrix
from warpack import PartialSVD


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("solver_type", [PartialSVD, NormalSVD])
def test_svd_replay_observes_changed_values_and_restoration(transpose, solver_type, tmp_path):
    rng = np.random.default_rng(912)
    a = rng.normal(size=(73, 29))
    a[rng.random(a.shape) > 0.35] = 0
    a[[3, 11], :] = 0
    a[:, [4, 12]] = 0
    if transpose:
        a = a.T.copy()
    sparse = sp.csr_matrix(a)
    wa = matrix(sparse)
    ncv = min(a.shape) if solver_type is NormalSVD else 32
    solver = solver_type(wa, 5, ncv=ncv, tol=1e-11)
    graph = solver.capture(100)
    original = sparse.data.copy()
    for values in (
        original,
        4 * original,
        original * rng.uniform(0.7, 1.3, len(original)),
        original,
    ):
        sparse.data[:] = values
        wa.values.assign(values)
        wp.capture_launch(graph)
        host = sparse.toarray()
        ref_u, ref_s, ref_vt = np.linalg.svd(host, full_matrices=False)
        w, u, v = solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T
        np.testing.assert_allclose(w, ref_s[:5], atol=1e-9, rtol=0)
        np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1e-10, rtol=0)
        np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1e-10, rtol=0)
        assert np.linalg.norm(host @ v - u * w, axis=0).max() < 1e-9
        assert np.linalg.norm(host.T @ u - v * w, axis=0).max() < 1e-9
        assert np.linalg.norm(u - ref_u[:, :5] @ (ref_u[:, :5].T @ u)) < 1e-9
        ref_v = ref_vt[:5].T
        assert np.linalg.norm(v - ref_v @ (ref_v.T @ v)) < 1e-9
        assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 5
        assert solver.core.status.numpy()[0] == 0
    dot = tmp_path / "svd-replay.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("transpose", [False, True])
def test_transpose_map_keeps_explicit_zero_slots(transpose):
    from warpack.svd import _Transpose

    # Include empty first/middle/last rows and explicit stored zeros that become
    # nonzero later. The topology mapping must preserve their slots.
    a = sp.csr_matrix(([0.0, 3.0, 0.0, 7.0], ([1, 1, 3, 3], [0, 4, 1, 4])), shape=(5, 7))
    if transpose:
        a = a.T.tocsr()
    coo = a.tocoo()
    wa = ws.bsr_from_triplets(
        *a.shape,
        wp.array(coo.row, dtype=wp.int32),
        wp.array(coo.col, dtype=wp.int32),
        wp.array(coo.data, dtype=wp.float64),
        prune_numerical_zeros=False,
    )
    # Sparse array capacity may exceed its logical dimensions.
    wa.offsets = wp.array(np.r_[wa.offsets.numpy(), [999, 999]], dtype=wp.int32)
    cached = _Transpose(wa)
    cached.refresh()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cached.refresh()
    for values in (a.data.copy(), np.array([2.0, -1.0, 8.0, 0.0])):
        a.data[:] = values
        wa.values.assign(values)
        wp.capture_launch(capture.graph)
        result = sp.csr_matrix(
            (
                cached.matrix.values.numpy(),
                cached.matrix.columns.numpy(),
                cached.matrix.offsets.numpy(),
            ),
            shape=a.T.shape,
        )
        np.testing.assert_array_equal(result.toarray(), a.T.toarray())
