"""Reject unsafe public storage metadata before any numerical dispatch."""

import numpy as np
import pytest
import warp as wp

from tests.test_solver import sparse
from warpack import BatchedSparseOperator, CuDSSInverse, SparseOperator


@pytest.mark.parametrize(
    "kind,bad",
    [
        (kind, bad)
        for kind in ("sparse", "batched", "cudss")
        for bad in ("short", "flat", "fp32", "cpu", "strided")
        if bad != "strided" or kind == "cudss"
    ],
)
@pytest.mark.parametrize("operand", ["input", "output"])
def test_invalid_storage_never_dispatches(kind, operand, bad, monkeypatch):
    n, width = 8, 2
    a = sparse(np.diag(np.arange(1.0, n + 1)))
    if kind == "cudss":
        # Metadata rejection must precede use of any native resources.
        op = object.__new__(CuDSSInverse)
        op.n, op.width, op.device = n, width, a.device
    else:
        op = (SparseOperator if kind == "sparse" else BatchedSparseOperator)(a)
    x = wp.ones((width, n), dtype=wp.float64, device=a.device)
    y = wp.empty_like(x)
    if bad == "short":
        invalid = wp.zeros((width, n - 1), dtype=wp.float64, device=a.device)
    elif bad == "flat":
        invalid = wp.zeros(width * n, dtype=wp.float64, device=a.device)
    elif bad == "fp32":
        invalid = wp.zeros((width, n), dtype=wp.float32, device=a.device)
    elif bad == "cpu":
        invalid = wp.zeros((width, n), dtype=wp.float64, device="cpu")
    else:
        invalid = wp.zeros((width, n * 2), dtype=wp.float64, device=a.device)[:, ::2]
    if operand == "input":
        x = invalid
    else:
        y = invalid

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid storage reached kernel dispatch")

    monkeypatch.setattr(wp, "launch", forbidden)
    monkeypatch.setattr(wp, "launch_tiled", forbidden)
    error = TypeError if bad == "fp32" else ValueError
    with pytest.raises(error):
        op.apply(x, y)


@pytest.mark.parametrize("kind", [SparseOperator, BatchedSparseOperator])
def test_strided_sparse_replay_preserves_padding(kind):
    n, width = 17, 3
    a = np.diag(np.arange(1.0, n + 1)) + np.eye(n, k=1) * 0.25
    op = kind(sparse(a))
    source = wp.zeros((width, 2 * n), dtype=wp.float64, device=op.device)
    storage = wp.full((width, 2 * n), 123.0, dtype=wp.float64, device=op.device)
    x, y = source[:, ::2], storage[:, ::2]
    op.apply(x, y)
    wp.synchronize_device(op.device)
    with wp.ScopedCapture(device=op.device) as capture:
        op.apply(x, y)
    for seed in [7, 19]:
        host = np.random.default_rng(seed).normal(size=source.shape)
        source.assign(host)
        wp.capture_launch(capture.graph)
        result = storage.numpy()
        np.testing.assert_allclose(result[:, ::2], host[:, ::2] @ a.T, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(result[:, 1::2], 123.0)
