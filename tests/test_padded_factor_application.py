"""Padded FSAI rows may change live counts during device-captured updates."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
import warp.sparse as ws

from warpack._coalesced_fsai import CoalescedFactorApplication
from warpack.batched_cg import RepeatedPreconditioner


def padded(n, dtype):
    a = ws.bsr_zeros(n, n, dtype, row_capacity=3)
    a.row_counts = wp.zeros(n, dtype=wp.int32)
    return a


def assign_padded(a, dense):
    n = len(dense)
    columns = np.full((n, 3), -1000000, dtype=np.int32)
    values = np.full((n, 3), np.nan)
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ids = np.flatnonzero(dense[i])
        counts[i] = len(ids)
        columns[i, : len(ids)] = ids
        values[i, : len(ids)] = dense[i, ids]
    a.columns.assign(columns.ravel())
    a.values.assign(values.ravel())
    a.row_counts.assign(counts)


@pytest.mark.parametrize("layout", ["rows", "columns"])
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("width", [7, 33])
def test_padded_factor_counts_and_values_refresh(layout, dtype, width, tmp_path):
    if layout == "rows":
        import inspect

        from warp.optim import linear

        if "max_batch_length" not in inspect.signature(linear.LinearOperator).parameters:
            pytest.skip("row-batched application requires the pinned Warp linear API")
    n = 37
    factors = SimpleNamespace(G=padded(n, wp.float32), GT=padded(n, wp.float32))
    g = np.eye(n)
    assign_padded(factors.G, g)
    assign_padded(factors.GT, g.T)
    offsets = wp.array(np.arange(width + 1, dtype=np.int32) * n, dtype=wp.int32)
    if layout == "columns":
        op = CoalescedFactorApplication(factors, width, n, factors.G.device, dtype)
    else:
        op = RepeatedPreconditioner(factors, width, n, offsets, factors.G.device, dtype=dtype)
    x = wp.zeros(width * n, dtype=dtype)
    y = wp.empty_like(x)
    op.matvec(x, y, x, 0.75, -0.25)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        op.matvec(x, y, x, 0.75, -0.25)
    rng = np.random.default_rng(139)
    for changed in (False, True, False):
        g = np.eye(n) + (np.diag(np.full(n - 1, 0.2), -1) if changed else 0)
        assign_padded(factors.G, g)
        assign_padded(factors.GT, g.T)
        xx = rng.normal(size=(width, n))
        yy = rng.normal(size=(width, n))
        x.assign(xx.ravel())
        y.assign(yy.ravel())
        wp.capture_launch(cap.graph)
        fg = g.astype(np.float32).astype(np.float64)
        bound = 2e-6 if dtype == wp.float32 else 2e-14
        np.testing.assert_allclose(
            x.numpy().reshape((width, n)),
            0.75 * ((xx @ fg.T) @ fg) - 0.25 * yy,
            rtol=bound,
            atol=bound,
        )
    path = tmp_path / "padded.dot"
    wp.capture_debug_dot_print(cap.graph, str(path))
    assert not any(x in path.read_text() for x in ("HOST", "HtoD", "DtoH"))
