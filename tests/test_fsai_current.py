"""Candidate dependency qualification; never silently replaces the release pin."""

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.experimental.fsai_current import load
from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner


@pytest.fixture(params=[False, True], ids=["current", "current-packed"])
def candidate_class(request):
    if not Path("build/current_fsai.py").exists() or not hasattr(linear, "FSAI"):
        pytest.skip("explicit current-FSAI candidate and pinned batched CG required")
    return load(packed=request.param).FSAI


def test_pinned_refit_failure_and_candidate_growth_order_fix(candidate_class):
    current = candidate_class
    b = np.sqrt(0.0179)
    a = matrix(np.array([[1, b, 0], [b, 1, 0.99], [0, 0.99, 1.0]]))
    old = linear.FSAI(a, max_row_size=3, pivot_floor=0.01, reuse_pattern=True)
    with pytest.raises(ValueError, match="SPD local supports"):
        old.update()
    new = current(a, max_row_size=3, pivot_floor=0.01, reuse_pattern=True)
    before = new.G.values.numpy().copy()
    new.update()
    np.testing.assert_allclose(new.G.values.numpy(), before, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("layout", ["rows", "columns"])
@pytest.mark.parametrize("deferred", [False, True])
def test_captured_update_and_cg_read_current_factor(layout, deferred, tmp_path, candidate_class):
    current = candidate_class
    n, width = 64, 7
    h = (
        np.diag(np.linspace(2, 3, n))
        + np.diag(np.full(n - 1, -0.4), 1)
        + np.diag(np.full(n - 1, -0.4), -1)
    )
    a = matrix(h)
    options = dict(
        max_row_size=8,
        max_step_size=3,
        reuse_pattern=True,
        defer_pattern=deferred,
        factor_dtype=wp.float32,
    )
    # Load update kernels on a separate instance: deferred selection must still
    # occur for the actual factor during the first graph replay.
    warm = current(a, **options)
    warm.update()
    factor = current(a, **options)
    pre = BatchedCGPreconditioner(
        a,
        width,
        tol=1e-12,
        maxiter=128,
        preconditioner=factor,
        fsai_layout=layout,
        matrix_layout="columns",
    )
    rhs = np.random.default_rng(139).normal(size=(width, n))
    x = wp.array(rhs, dtype=wp.float64)
    y = wp.empty_like(x)
    pre.apply(x, y)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        factor.update()
        pre.reset_stats()
        pre.apply(x, y)
    if deferred:
        assert factor.pattern_ready.numpy()[0] == 0
    addresses = tuple(v.ptr for v in (factor.G.values, factor.GT.values))
    for scale in (1.0, 1.5, 0.8):
        a.values.assign(matrix(h * scale).values.numpy())
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(factor.update_status.numpy(), 0)
        assert pre.stats.numpy()[2] == 0
        if deferred:
            assert factor.pattern_ready.numpy()[0] == 1
        assert addresses == tuple(v.ptr for v in (factor.G.values, factor.GT.values))
        np.testing.assert_allclose(
            y.numpy().T, la.solve(h * scale, rhs.T, assume_a="pos"), rtol=1e-10, atol=1e-11
        )
    # Failed updates preserve the preceding factor and report device failure.
    previous = factor.G.values.numpy().copy()
    invalid = h.copy()
    invalid[0, 0] = -1
    a.values.assign(matrix(invalid).values.numpy())
    wp.capture_launch(cap.graph)
    assert factor.update_status.numpy()[1] != 0  # invalid diagonals, not topology
    np.testing.assert_array_equal(factor.G.values.numpy(), previous)
    a.values.assign(matrix(h).values.numpy())
    wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(factor.update_status.numpy(), 0)
    assert pre.stats.numpy()[2] == 0
    np.testing.assert_allclose(
        y.numpy().T, la.solve(h, rhs.T, assume_a="pos"), rtol=1e-10, atol=1e-11
    )
    path = tmp_path / "update-cg.dot"
    wp.capture_debug_dot_print(cap.graph, str(path))
    assert not any(s in path.read_text() for s in ("HOST", "DtoH", "HtoD"))


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_captured_refit_and_fem_eigensolve(
    consistent, deferred, monkeypatch, tmp_path, candidate_class
):
    from benchmarks.applications.fem_shapes import ShapeReplay, shape_family

    current = candidate_class

    def constructor(a, **options):
        options.pop("apply_lanes")  # Removed by the candidate dependency.
        return current(a, defer_pattern=deferred, **options)

    monkeypatch.setattr(linear, "FSAI", constructor)
    states = shape_family(2, 6, consistent)
    driver = ShapeReplay(states, 12, steps=None)
    driver.graph = driver.solver.capture(200, initializer=driver.factor.update)
    for repeat in range(2):
        for i, (_, problem, _) in enumerate(states):
            # Upload changed K/M outside capture; factor update and eigen solve
            # share one graph. No eager update call may mask a stale factor.
            driver.update(problem, factor=False)
            driver.solve(warm=i > 0)
            np.testing.assert_array_equal(driver.factor.update_status.numpy(), 0)
            quality = driver.quality(problem)
            assert quality["accepted"], quality
    path = tmp_path / "refit-fem.dot"
    wp.capture_debug_dot_print(driver.graph, str(path))
    assert not any(s in path.read_text() for s in ("HOST", "DtoH", "HtoD"))
