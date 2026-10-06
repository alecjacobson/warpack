import inspect

import numpy as np
import pytest
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from warpack import CGInverse

pytestmark = pytest.mark.skipif(
    "block_jacobi_sequential" not in inspect.getsource(linear.preconditioner),
    reason="requires upstream Warp block-Jacobi support",
)


def matrix(coupling):
    rng = np.random.default_rng(37)
    blocks = []
    rows, cols = [], []
    for i in range(8):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        blocks.append(q @ np.diag([1.0, 100.0, 10000.0]) @ q.T)
        rows.append(i)
        cols.append(i)
        if i > 0 and coupling:
            for row, col in [(i - 1, i), (i, i - 1)]:
                blocks.append(-0.01 * np.eye(3))
                rows.append(row)
                cols.append(col)
    a = ws.bsr_from_triplets(
        8,
        8,
        wp.array(rows, dtype=wp.int32),
        wp.array(cols, dtype=wp.int32),
        wp.array(blocks, dtype=wp.mat33d),
    )
    dense = np.zeros((24, 24))
    for r, c, v in zip(rows, cols, blocks):
        dense[3 * r : 3 * r + 3, 3 * c : 3 * c + 3] = v
    return a, dense


@pytest.mark.parametrize("coupling", [False, True])
def test_block_jacobi_cg_captured_rhs_replay(coupling):
    a, dense = matrix(coupling)
    inverse = CGInverse(a, tol=1e-13, maxiter=100, preconditioner="block_jacobi_sequential")
    rhs = wp.array(np.ones((1, 24)), dtype=wp.float64)
    output = wp.empty_like(rhs)
    inverse.apply(rhs, output)
    with wp.ScopedCapture() as cap:
        inverse.reset_stats()
        inverse.apply(rhs, output)
    for seed in [11, 23]:
        b = np.random.default_rng(seed).normal(size=(1, 24))
        wp.copy(rhs, wp.array(b, dtype=wp.float64))
        wp.capture_launch(cap.graph)
        answer = output.numpy()[0]
        np.testing.assert_allclose(answer, np.linalg.solve(dense, b[0]), atol=2e-11, rtol=2e-11)
        assert np.linalg.norm(dense @ answer - b[0]) / np.linalg.norm(b) < 1e-11
        total, maximum, failed, calls = inverse.stats.numpy()
        assert total == maximum and failed == 0 and calls == 1
        if not coupling:
            assert total <= 2
