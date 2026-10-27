"""A first-metric shortcut must prove rank, including after failed replays."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.first_metric_certificate import FirstMetricCertificate
from benchmarks.graph_audit import graph_summary


def correlation(n):
    a = np.random.default_rng(171).normal(size=(n, 2 * n))
    g = a @ a.T / (2 * n) + 0.1 * np.eye(n)
    d = np.sqrt(np.diag(g))
    g = g / d[:, None] / d[None, :]
    return (g + g.T) * 0.5


@pytest.mark.parametrize("n", [8, 48, 104, 208, 408, 512])
def test_finite_full_rank_bounds_strides_and_replay(n, tmp_path):
    original = correlation(n)
    store = wp.full((n, n + 3), 17.0, dtype=wp.float64)
    g = store[:, :n]
    cert = FirstMetricCertificate(g)
    g.assign(original)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    for source, expected in [
        (original, True),
        (np.zeros((n, n)), False),
        (np.eye(n), True),
        (original, True),
    ]:
        g.assign(source)
        wp.capture_launch(capture.graph)
        assert bool(cert.accepted.numpy()[0]) == expected
        np.testing.assert_array_equal(g.numpy(), source)
        assert np.all(store.numpy()[:, n:] == 17.0)
        if expected:
            delta, lower, upper, _ = cert.diagnostic.numpy()
            values = la.eigvalsh(source)
            assert 0 < lower <= values[0]
            assert values[-1] <= upper
            assert lower > cert.rank_tol * upper
            r = cert.inverse.numpy().astype(np.longdouble)
            actual = r @ source.astype(np.longdouble) @ r.T - np.eye(n, dtype=np.longdouble)
            assert float(np.max(np.sum(abs(actual), axis=1))) <= delta <= 0.25
    dot = tmp_path / "certificate.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    graph = graph_summary(dot)
    assert not graph["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in graph["memcpy_kinds"])


@pytest.mark.parametrize("gap", [1e-2, 1e-5, 1e-7, 1e-9, 1e-11, 1e-12, 0.0, -1e-6])
def test_near_dependent_pair_never_accepts_discarded_direction(gap):
    original = np.array([[1.0, 1.0 - gap], [1.0 - gap, 1.0]])
    cert = FirstMetricCertificate(wp.array(original, dtype=wp.float64))
    cert.apply()
    accepted = bool(cert.accepted.numpy()[0])
    if accepted:
        values = la.eigvalsh(original)
        assert values[0] > cert.rank_tol * values[-1]
        _, lower, upper, _ = cert.diagnostic.numpy()
        assert lower <= values[0] and upper >= values[-1]
    if gap >= 1e-5:
        assert accepted
    if gap <= 1e-12:
        assert not accepted


@pytest.mark.parametrize("kind", ["nan", "inf", "indefinite", "nonsymmetric"])
def test_reject_invalid_and_recover_on_same_graph(kind):
    g = wp.array(np.eye(8), dtype=wp.float64)
    cert = FirstMetricCertificate(g)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    broken = np.eye(8)
    if kind == "nan":
        broken[0, 0] = np.nan
    elif kind == "inf":
        broken[0, 0] = np.inf
    elif kind == "indefinite":
        broken[0, 1] = broken[1, 0] = 2.0
    else:
        broken[0, 1] = 0.1
    for source, expected in [(broken, False), (np.eye(8), True), (correlation(8), True)]:
        g.assign(source)
        wp.capture_launch(capture.graph)
        assert bool(cert.accepted.numpy()[0]) == expected
        np.testing.assert_array_equal(g.numpy(), source)


@pytest.mark.parametrize(
    "tol, expected", [(0.1, True), (0.9, True), (1.0 - 1e-14, False), (1.0, False)]
)
def test_conservative_rank_threshold(tol, expected):
    cert = FirstMetricCertificate(wp.array(np.eye(8), dtype=wp.float64), rank_tol=tol)
    cert.apply()
    assert bool(cert.accepted.numpy()[0]) == expected


@pytest.mark.parametrize("n", [8, 104, 208])
@pytest.mark.parametrize("smallest", [1.0, 1e-3, 1e-7, 1e-11, 1e-13, 0.0, -1e-5])
def test_random_rotated_spectrum_has_no_false_rank_certificate(n, smallest):
    q = la.qr(np.random.default_rng(171 + n).normal(size=(n, n)))[0]
    spectrum = np.linspace(0.5, 1.0, n)
    spectrum[0] = smallest
    g = (q * spectrum) @ q.T
    d = np.sqrt(np.diag(g))
    g = g / d[:, None] / d[None, :]
    g = (g + g.T) * 0.5
    cert = FirstMetricCertificate(wp.array(g, dtype=wp.float64))
    cert.apply()
    accepted = bool(cert.accepted.numpy()[0])
    values = la.eigvalsh(g)
    if accepted:
        delta, lower, upper, _ = cert.diagnostic.numpy()
        assert 0 < lower <= values[0] and values[-1] <= upper
        assert values[0] > cert.rank_tol * values[-1]
        r = cert.inverse.numpy().astype(np.longdouble)
        error = r @ g.astype(np.longdouble) @ r.T - np.eye(n, dtype=np.longdouble)
        assert np.max(np.sum(abs(error), axis=1)) <= delta <= 0.25
    if smallest >= 1e-3:
        assert accepted
    if smallest <= 1e-13:
        assert not accepted


@wp.kernel(enable_backward=False)
def inaccurate_successful_factor(
    g: wp.array2d[wp.float64],
    factor: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    work: wp.array2d[wp.float64],
):
    block, lane = wp.tid()
    for i in range(lane, g.shape[0], 256):
        for j in range(g.shape[1]):
            factor[i, j] = wp.float64(i == j)
    if lane == 0:
        status[0] = 0


def test_successful_factor_status_alone_cannot_accept_bad_transform(monkeypatch):
    from benchmarks.experimental import first_metric_certificate as module

    monkeypatch.setattr(module, "global_cholesky", inaccurate_successful_factor)
    g = wp.array(np.eye(8), dtype=wp.float64)
    cert = FirstMetricCertificate(g)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    bad = np.eye(8)
    bad[0, 1] = bad[1, 0] = 0.9
    for source, expected in [(np.eye(8), True), (bad, False), (np.eye(8), True)]:
        g.assign(source)
        wp.capture_launch(capture.graph)
        assert cert.status.numpy()[0] == 0
        assert bool(cert.accepted.numpy()[0]) == expected


@pytest.mark.parametrize("tol", [-1.0, np.nan, np.inf])
def test_invalid_rank_threshold_rejects(tol):
    cert = FirstMetricCertificate(wp.array(np.eye(8), dtype=wp.float64), rank_tol=tol)
    cert.apply()
    assert cert.accepted.numpy()[0] == 0
