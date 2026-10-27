"""An auxiliary identity must never turn padding into physical search rank."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.padded_first_metric import PaddedFirstMetricCertificate
from benchmarks.graph_audit import graph_summary


def correlated(n):
    a = np.random.default_rng(171).normal(size=(n, n))
    a = a @ a.T / n + 2 * np.eye(n)
    d = np.sqrt(np.diag(a))
    return a / (d[:, None] * d[None, :])


@pytest.mark.parametrize("n", [8, 104, 408])
def test_permuted_padding_changed_rank_and_unchanged_input(n, tmp_path):
    full = correlated(n)
    source = wp.array(full, dtype=wp.float64)
    cert = PaddedFirstMetricCertificate(source)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    for count in [n, n - 3, n // 2, 1, 0, n]:
        mask = np.zeros(n, dtype=bool)
        mask[np.random.default_rng(count).permutation(n)[:count]] = True
        g = full.copy()
        g[~mask, :] = 0
        g[:, ~mask] = 0
        source.assign(g)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(source.numpy(), g)
        np.testing.assert_array_equal(cert.active.numpy().astype(bool), mask)
        assert cert.count.numpy()[0] == count
        assert cert.accepted.numpy()[0] == int(count > 0)
        if count:
            r = cert.inverse.numpy()
            physical = (r @ g) @ r.T
            np.testing.assert_allclose(
                physical, np.diag(mask.astype(float)), atol=2e-13, rtol=2e-13
            )
            assert np.count_nonzero(la.eigvalsh(g) > 1e-12 * la.norm(g, 2)) == count
            assert np.all(physical[~mask] == 0)
            assert np.all(physical[:, ~mask] == 0)
            delta, lower, upper, error = cert.certificate.diagnostic.numpy()
            actual = la.eigvalsh(g[np.ix_(mask, mask)])
            assert 0 <= delta <= 0.25 and error >= 0
            assert lower <= actual[0] and upper >= actual[-1]
    dot = tmp_path / "padding.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    audit = graph_summary(dot)
    assert not audit["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "inf",
        "asymmetric",
        "zero_coupling",
        "negative",
        "unnormalized",
        "dependent",
        "near_rank_cutoff",
    ],
)
def test_uncertainty_falls_back_and_recovers(kind):
    n = 24
    good = correlated(n)
    good[::3] = 0
    good[:, ::3] = 0
    source = wp.array(good, dtype=wp.float64)
    cert = PaddedFirstMetricCertificate(source)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    broken = good.copy()
    if kind in ("nan", "inf"):
        broken[1, 1] = np.nan if kind == "nan" else np.inf
    elif kind == "asymmetric":
        broken[1, 2] += 0.1
    elif kind == "zero_coupling":
        broken[0, 1] = broken[1, 0] = 1e-30
    elif kind == "negative":
        broken[1, 1] = -1
    elif kind == "unnormalized":
        broken *= 0.1
    else:
        broken = np.eye(n)
        broken[::3] = 0
        broken[:, ::3] = 0
        broken[1, 2] = broken[2, 1] = 1 if kind == "dependent" else 1 - 1e-13
    for matrix, expected in [(broken, 0), (good, 1)]:
        source.assign(matrix)
        wp.capture_launch(capture.graph)
        assert cert.accepted.numpy()[0] == expected
        np.testing.assert_array_equal(source.numpy(), matrix)


@pytest.mark.parametrize("condition", [1.0, 1e4, 1e8, 1e12, 1e14])
def test_rotated_active_blocks_certificate_is_conservative(condition):
    n, rank = 24, 13
    source = wp.zeros((n, n), dtype=wp.float64)
    cert = PaddedFirstMetricCertificate(source)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    for seed in range(4):
        rng = np.random.default_rng(171 + seed)
        q = la.qr(rng.normal(size=(rank, rank)))[0]
        active_g = (q * np.geomspace(1 / condition, 1, rank)) @ q.T
        d = np.sqrt(np.diag(active_g))
        active_g /= d[:, None] * d[None, :]
        active_g = (active_g + active_g.T) * 0.5
        indices = rng.permutation(n)[:rank]
        g = np.zeros((n, n))
        g[np.ix_(indices, indices)] = active_g
        source.assign(g)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(source.numpy(), g)
        if cert.accepted.numpy()[0]:
            delta, lower, upper, _ = cert.certificate.diagnostic.numpy()
            actual = la.eigvalsh(active_g)
            assert lower > 1e-12 * upper
            assert lower <= actual[0] + 1e-14 and upper >= actual[-1]
            r = cert.inverse.numpy().astype(np.longdouble)
            auxiliary = g.copy()
            absent = np.ones(n, dtype=bool)
            absent[indices] = False
            auxiliary[np.diag_indices(n)] += absent
            error = r @ auxiliary.astype(np.longdouble) @ r.T - np.eye(n, dtype=np.longdouble)
            assert np.max(np.sum(abs(error), axis=1)) <= delta <= 0.25
            assert actual[0] > 1e-12 * actual[-1]
