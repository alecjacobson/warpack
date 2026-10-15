"""Block-product experiments must preserve FP64 results and captured inputs."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws

from benchmarks.experimental.block_kernels import BatchedSparseOperator, UpstreamSparseOperator
from benchmarks.experimental.block_lobpcg import gram
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("chunk", [256, 1024, 4096])
def test_rectangular_partial_tiles_and_changed_gram_inputs(chunk):
    rng = np.random.default_rng(713)
    a = rng.normal(size=(13, 4157))
    b = rng.normal(size=(19, 4157))
    x, y = wp.array(a, dtype=wp.float64), wp.array(b, dtype=wp.float64)
    out = wp.empty((13, 19), dtype=wp.float64)
    gram(x, y, out, chunk=chunk)
    wp.synchronize()  # Complete setup/JIT warmup, as the solver's capture() does.
    with wp.ScopedCapture() as capture:
        gram(x, y, out, chunk=chunk)
    for scale in [1.0, -1e-30, 1e30]:
        x.assign(a * scale)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(out.numpy() / scale, a @ b.T, atol=2e-11, rtol=2e-12)


@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("operator", [UpstreamSparseOperator, BatchedSparseOperator])
def test_upstream_sparse_block_products_replay(block, operator):
    rng = np.random.default_rng(26)
    a = sp.random(93, 93, density=0.2, random_state=rng, format="csr")
    a = a + a.T
    if block == 1:
        wa = matrix(a)
    else:
        b = a.tobsr(blocksize=(3, 3))
        wa = ws.bsr_from_triplets(
            31,
            31,
            wp.array(np.repeat(np.arange(31), np.diff(b.indptr)), dtype=wp.int32),
            wp.array(b.indices, dtype=wp.int32),
            wp.array(b.data, dtype=wp.mat33d),
        )
    op = operator(wa)
    x, y = wp.zeros((7, 93), dtype=wp.float64), wp.zeros((7, 93), dtype=wp.float64)
    op.apply(x, y)
    with wp.ScopedCapture() as capture:
        op.apply(x, y)
    for _ in range(2):
        source = rng.normal(size=x.shape)
        x.assign(source)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(y.numpy(), (a @ source.T).T, atol=1e-12, rtol=1e-12)
