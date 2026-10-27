"""Factor application preserves coefficients, aliases, tails and changed replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
import warp.sparse as ws

from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack._coalesced_fsai import CoalescedFactorApplication


@pytest.mark.parametrize("factor_dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("working_dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("width", [1, 7, 24, 33])
@pytest.mark.parametrize("alias", [False, True])
def test_coalesced_factor_replay(factor_dtype, working_dtype, width, alias, tmp_path):
    rng = np.random.default_rng(71)
    n = 113
    g = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.07) * 0.1 + np.eye(n)
    factors = SimpleNamespace(
        G=ws.bsr_copy(matrix(g), scalar_type=factor_dtype),
        GT=ws.bsr_copy(matrix(g.T), scalar_type=factor_dtype),
    )
    operator = CoalescedFactorApplication(factors, width, n, factors.G.device, working_dtype)
    x = wp.zeros(width * n, dtype=working_dtype)
    y = wp.empty_like(x)
    out = x if alias else wp.empty_like(x)
    operator.matvec(x, y, out, 0.75, -0.25)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        operator.matvec(x, y, out, 0.75, -0.25)
    dtype = np.float32 if working_dtype == wp.float32 else np.float64
    fg = g.astype(np.float32 if factor_dtype == wp.float32 else np.float64).astype(dtype)
    for trial in range(2):
        if trial:
            # Existing captured factor buffers may be updated without recapture.
            for factor in (factors.G, factors.GT):
                factor.values.assign(factor.values.numpy() * 0.5)
            fg *= 0.5
        xx, yy = [rng.normal(size=(width, n)).astype(dtype) for _ in range(2)]
        x.assign(xx.ravel())
        y.assign(yy.ravel())
        wp.capture_launch(capture.graph)
        expected = 0.75 * ((xx @ fg.T) @ fg) - 0.25 * yy
        bound = 2e-6 if working_dtype == wp.float32 else 2e-14
        np.testing.assert_allclose(
            out.numpy().reshape((width, n)), expected, rtol=bound, atol=bound
        )
        if not alias:
            np.testing.assert_array_equal(x.numpy(), xx.ravel())
    y.fill_(float("nan"))
    x.assign(xx.ravel())
    operator.matvec(x, y, out, 1.0, 0.0)
    np.testing.assert_allclose(
        out.numpy().reshape((width, n)), (xx @ fg.T) @ fg, rtol=bound, atol=bound
    )
    dot = tmp_path / "coalesced.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
