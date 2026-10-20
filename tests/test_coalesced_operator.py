"""Independent sparse-operator checks for the experimental RHS layout."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws

from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.coalesced_operator import CoalescedSparseApplication
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("width", [1, 7, 24, 33])
@pytest.mark.parametrize("alias", [False, True])
def test_coalesced_sparse_replay(dtype, block, width, alias, tmp_path):
    rng = np.random.default_rng(415)
    n = 117
    host = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.04) + np.eye(n)
    if block == 1:
        a = ws.bsr_copy(matrix(host), scalar_type=dtype)
    else:
        b = sp.csr_matrix(host).tobsr(blocksize=(3, 3))
        a = ws.bsr_from_triplets(
            n // 3,
            n // 3,
            wp.array(np.repeat(np.arange(n // 3), np.diff(b.indptr)), dtype=wp.int32),
            wp.array(b.indices, dtype=wp.int32),
            wp.array(b.data, dtype=wp.types.matrix(shape=(3, 3), dtype=dtype)),
        )
    op = CoalescedSparseApplication(a, width)
    x, y = [wp.zeros(width * n, dtype=dtype) for _ in range(2)]
    out = x if alias else wp.empty_like(x)
    op.matvec(x, y, out, 0.75, -0.25)
    wp.synchronize()
    with wp.ScopedCapture() as captured:
        op.matvec(x, y, out, 0.75, -0.25)
    ndtype = np.float32 if dtype == wp.float32 else np.float64
    ah = host.astype(ndtype)
    for trial in range(2):
        if trial:
            a.values.assign(a.values.numpy() * 0.5)
            ah *= 0.5
        xx, yy = [rng.normal(size=(width, n)).astype(ndtype) for _ in range(2)]
        x.assign(xx.ravel())
        y.assign(yy.ravel())
        wp.capture_launch(captured.graph)
        bound = 3e-6 if dtype == wp.float32 else 2e-14
        np.testing.assert_allclose(
            out.numpy().reshape((width, n)), 0.75 * (xx @ ah.T) - 0.25 * yy, rtol=bound, atol=bound
        )
        if not alias:
            np.testing.assert_array_equal(x.numpy(), xx.ravel())
    x.assign(xx.ravel())
    y.fill_(float("nan"))
    op.matvec(x, y, out, 1.0, 0.0)
    np.testing.assert_allclose(out.numpy().reshape((width, n)), xx @ ah.T, rtol=bound, atol=bound)
    dot = tmp_path / "operator.dot"
    wp.capture_debug_dot_print(captured.graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert not audit["memcpy_kinds"]
