"""Guarded QR reuse must preserve original equations and changed-input recovery."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.experimental.metric_qr_reuse import MetricQRReuseBlock, metric_defect
from benchmarks.upstream.adapters import matrix
from tests import test_certified_whitening, test_first_whitening, test_triangular_rotate
from warpack import SparseOperator


@pytest.mark.parametrize("tolerance", [0.0, -1.0, 1e-8, np.nan, np.inf])
def test_invalid_tolerance_before_allocation(tolerance):
    class Candidate(MetricQRReuseBlock):
        _metric_reuse_tolerance = tolerance

    with pytest.raises(ValueError, match="metric-defect"):
        Candidate(None, 1)


def test_computed_defect_bound_and_nonfinite_detection():
    n = 27
    rng = np.random.default_rng(183)
    base = rng.normal(size=(n, n))
    data = wp.array(np.eye(n), dtype=wp.float64)
    bound = wp.zeros(1, dtype=wp.float64)
    invalid = wp.zeros(1, dtype=wp.int32)
    wp.launch(metric_defect, n, [data, bound, invalid])
    with wp.ScopedCapture() as capture:
        bound.zero_()
        invalid.zero_()
        wp.launch(metric_defect, n, [data, bound, invalid])
    for scale in [1.0, 1e-14, 1e80, 0.0]:
        host = np.eye(n) + scale * base
        data.assign(host)
        wp.capture_launch(capture.graph)
        error = np.abs(host - np.eye(n))
        reference = max(error.sum(axis=0).max(), error.sum(axis=1).max())
        actual = bound.numpy()[0]
        assert invalid.numpy()[0] == 0
        assert reference <= actual * (1 + 1e-14)
        assert actual <= reference * (1 + 1e-12)
    host[3, 2] = np.nan
    data.assign(host)
    wp.capture_launch(capture.graph)
    assert invalid.numpy()[0] == 1


@pytest.mark.parametrize("tolerance", [1e-14, 1e-13, 1e-12])
def test_reuse_fallback_and_failure_recovery_on_one_graph(tolerance):
    class Candidate(MetricQRReuseBlock):
        _metric_reuse_tolerance = tolerance

    width, n = 24, 103
    diagonal = np.linspace(1, 4, n)
    mass_diagonal = np.linspace(1, 2, n)
    mass = matrix(sp.diags(mass_diagonal, format="csr"))
    solver = Candidate(
        SparseOperator(matrix(sp.diags(diagonal, format="csr"))),
        4,
        width,
        mass=SparseOperator(mass),
        gram_chunk=4096,
        reuse_operator_images=True,
        symmetric_products=True,
    )
    solver._image_reuse_active = True
    exact = np.eye(n)[:width] / np.sqrt(mass_diagonal)
    solver.sx.assign(exact)
    solver._image_x.assign(exact * diagonal)
    solver.orthogonalize_x()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver.metric_reuse_usage.zero_()
        solver.orthogonalize_x()
    rng = np.random.default_rng(183)
    for kind in [
        "reuse",
        "random",
        "changed_mass",
        "scaled",
        "zero",
        "negative",
        "nonfinite",
        "reuse",
    ]:
        md = mass_diagonal * (1.25 if kind == "changed_mass" else -1 if kind == "negative" else 1)
        mass.values.assign(md)
        source = exact.copy()
        if kind == "random":
            source = rng.normal(size=(width, n))
        elif kind == "scaled":
            source *= np.geomspace(1e-60, 1e60, width)[:, None]
        elif kind == "zero":
            source[:] = 0
        elif kind == "nonfinite":
            source[0, 0] = np.nan
        solver.sx.assign(source)
        solver._image_x.assign(source * diagonal)
        wp.capture_launch(capture.graph)
        healthy = kind not in ("zero", "negative", "nonfinite")
        assert (solver.status.numpy()[0] == 0) == healthy
        np.testing.assert_array_equal(
            solver.metric_reuse_usage.numpy(), [1, 0] if kind == "reuse" else [0, 1]
        )
        if healthy:
            x, ax, mx = solver.sx.numpy(), solver._image_x.numpy(), solver.bx.numpy()
            assert np.linalg.norm(x @ (md * x).T - np.eye(width)) < 1e-10
            np.testing.assert_allclose(ax, x * diagonal, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(mx, x * md, atol=1e-12, rtol=1e-12)
            if kind == "reuse":
                np.testing.assert_array_equal(x, source)
    assert solver._ritz_qr_reuse.eligible


@pytest.mark.parametrize("tolerance", [1e-14, 1e-13, 1e-12])
@pytest.mark.parametrize("cached", [False, True])
def test_coupled_scaled_complete_stage(tolerance, cached):
    class Candidate(MetricQRReuseBlock):
        _metric_reuse_tolerance = tolerance

    test_triangular_rotate.test_orthogonalization_with_changed_mass_scaling_and_cached_images(
        104, cached, Candidate
    )


@pytest.mark.parametrize("history", [False, True])
def test_changed_coupled_constraints_clusters(monkeypatch, tmp_path, history):
    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", MetricQRReuseBlock)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_custom_mass_retains_original_qr(wrapped):
    class CustomMass(SparseOperator):
        requires_static_outer_capture = True

    n, width = 103, 24
    operator = SparseOperator(matrix(sp.eye(n, format="csr")))
    mass = (SparseOperator if wrapped else CustomMass)(matrix(sp.eye(n, format="csr")))
    if wrapped:
        original = mass.apply
        mass.apply = lambda x, y: original(x, y)
    solver = MetricQRReuseBlock(operator, 4, width, mass=mass)
    assert not solver._ritz_qr_reuse.eligible
    source = np.random.default_rng(183).normal(size=(width, n))
    solver.sx.assign(source)
    solver.orthogonalize_x()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver.orthogonalize_x()
    solver.sx.assign(source)
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] == 0
    np.testing.assert_array_equal(solver.metric_reuse_usage.numpy(), [0, 0])
    x = solver.sx.numpy()
    assert np.linalg.norm(x @ x.T - np.eye(width)) < 1e-10
