"""Exact active-RHS partitioning must preserve accuracy, alias safety and work counts."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.experimental.batched_cg import CapturedActiveCG, CapturedPartitionedCG
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner


@pytest.mark.parametrize("dtype,tol", [(wp.float64, 1e-11), (wp.float32, 1e-5)])
@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_exact_chunks_match_whole_block_with_fsai_and_shifted_alias(
    dtype, tol, layout, tmp_path, monkeypatch
):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    n, width = 97, 13
    a = sp.diags(
        [-np.ones(n - 1), np.linspace(2.1, 5, n), -np.ones(n - 1)], [-1, 0, 1], format="csr"
    )
    wa = matrix(a)
    factor = linear.FSAI(wa, max_row_size=8, factor_dtype=wp.float32)
    inverse = BatchedCGPreconditioner(
        wa,
        width,
        tol=tol * 0.1,
        atol=0,
        maxiter=100,
        preconditioner=factor,
        working_dtype=dtype,
        fsai_layout=layout,
        matrix_layout=layout,
    )
    split, exact = CapturedPartitionedCG(inverse), CapturedActiveCG(inverse)
    split.reset_stats()
    exact.reset_stats()
    assert list(split.adapters) == [1, 2, 4, 8]
    assert sum(split.adapters) < 2 * width
    assert all(item.graph.graph_exec is not None for item in split.adapters.values())
    rng = np.random.default_rng(146)
    total = 0
    for active in (13, 1, 7, 2, 12, 13):
        source = rng.normal(size=(active, n)) * np.geomspace(1e-5, 1e5, active)[:, None]
        source[0] = 0
        buffers = wp.zeros((active + 1, n), dtype=wp.float64)
        wp.copy(buffers[:active], wp.array(source, dtype=wp.float64))
        whole = wp.array(source, dtype=wp.float64)
        exact.apply(whole, whole)
        split.apply(buffers[:active], buffers[1:])
        actual = buffers[1:].numpy()
        residual = np.linalg.norm((a @ actual.T).T - source, axis=1)
        assert np.all(residual <= tol * np.linalg.norm(source, axis=1))
        np.testing.assert_allclose(actual, whole.numpy(), rtol=tol * 10, atol=tol * 1e-6)
        total += active
    stats = split.collect_stats().numpy()
    assert stats[2] == 0 and stats[3] == total
    split.reset_stats()
    assert np.all(split.collect_stats().numpy() == 0)
    for width, item in split.adapters.items():
        assert item.inverse.single_preconditioner is factor
        path = tmp_path / f"partition-{width}.dot"
        wp.capture_debug_dot_print(item.graph, str(path))
        graph = graph_summary(path)
        assert not graph["node_types_including_conditional_body"].get("HOST", 0)
        assert all("DtoD" in kind for kind in graph["memcpy_kinds"])

    # No numerical workspace allocation is permitted in apply(). Array views
    # and the graph enqueue itself may create ordinary Python wrapper objects.
    def forbidden(*args, **kwargs):
        raise AssertionError("device allocation during replay")

    monkeypatch.setattr(wp, "empty", forbidden)
    monkeypatch.setattr(wp, "empty_like", forbidden)
    split.apply(whole, whole)
    wp.synchronize()


def test_partitioned_truncated_statistics_and_validation():
    n, width = 43, 11
    a = sp.diags(
        [-np.ones(n - 1), np.linspace(2.1, 8, n), -np.ones(n - 1)], [-1, 0, 1], format="csr"
    )
    inverse = BatchedCGPreconditioner(matrix(a), width, tol=1e-13, atol=0, maxiter=1)
    split = CapturedPartitionedCG(inverse)
    split.reset_stats()
    rng = np.random.default_rng(1461)
    source = rng.normal(size=(width, n))
    source[0] = 0
    rhs = wp.array(source, dtype=wp.float64)
    split.apply(rhs, rhs)
    stats = split.collect_stats().numpy()
    assert stats[3] == width and stats[2] == width - 1
    # Work is charged only to actual RHSs, including the explicit zero RHS.
    assert stats[0] == width and stats[1] == 1
    with pytest.raises(ValueError):
        split.apply(rhs[:1], rhs[:2])
    with pytest.raises(ValueError):
        CapturedPartitionedCG(inverse, max_width=width + 1)
    with pytest.raises(ValueError):
        CapturedPartitionedCG(inverse, max_width=0)
    with pytest.raises(ValueError):
        CapturedPartitionedCG(inverse, max_width=2.5)
