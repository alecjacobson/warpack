"""Padding certificate fallback, temporal scratch aliasing and strided storage."""

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.padded_first_metric import PaddedFirstMetricCertificate
from benchmarks.experimental.padded_first_whitening import PaddedFirstWhiteningBlock
from tests.test_first_whitening import AllSizePaddedWhitening, failed_factor
from warpack import BlockLOBPCG
from warpack.block import Identity


def test_strided_certificate_scratch_and_sentinels():
    n = 104
    stores = [wp.full((n, n + 5), 17.0, dtype=wp.float64) for _ in range(6)]
    source, *scratch = [a[:, :n] for a in stores]
    cert = PaddedFirstMetricCertificate(source, storage=scratch)
    good = np.eye(n)
    good[::3, ::3] = 0
    source.assign(good)
    cert.apply()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        cert.apply()
    for a in [good, np.eye(n), np.zeros((n, n)), good]:
        source.assign(a)
        wp.capture_launch(capture.graph)
        assert cert.accepted.numpy()[0] == int(np.any(a))
        np.testing.assert_array_equal(source.numpy(), a)
        for store in stores:
            assert np.all(store.numpy()[:, n:] == 17.0)


def test_failed_padded_attempt_uses_untouched_gram_and_outer_status(monkeypatch):
    from benchmarks.experimental import first_metric_certificate

    monkeypatch.setattr(first_metric_certificate, "global_cholesky", failed_factor)
    source = np.random.default_rng(2171).normal(size=(8, 32))
    source[::3] = 0
    results = []
    for solver_type in [BlockLOBPCG, AllSizePaddedWhitening]:
        solver = solver_type(Identity(32, wp.get_device()), 2, 4)
        solver.su.assign(source)
        solver.whiten(solver.su, solver.u1, normalize=True)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            solver.status.zero_()
            solver.whiten(solver.su, solver.u1, normalize=True)
        solver.su.assign(source)
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == 0
        assert solver.rank.numpy()[0] == 5
        results.append(solver.u1.numpy())
        if solver_type is AllSizePaddedWhitening:
            np.testing.assert_array_equal(solver._first_whitening_usage.numpy(), [0, 1, 0])
    np.testing.assert_array_equal(results[0], results[1])


def test_padded_and_second_whitening_share_only_three_dense_buffers():
    solver = PaddedFirstWhiteningBlock(
        Identity(400, wp.get_device()), 4, 52, reuse_operator_images=True, search_bucket_size=4
    )
    assert solver._first_whitening_storage is solver._second_whitening_storage
    for branch in [solver, *solver._search.buckets]:
        if branch._first_whitener is None:
            continue
        wrapper = branch._first_whitener.certificate
        cert = wrapper.certificate
        assert wrapper.original.ptr == branch.metric.g.ptr
        assert cert.g.ptr == solver._second_whitening_storage[0].ptr
        assert cert.factor.ptr == solver._second_whitening_storage[1].ptr
        assert cert.work.ptr == cert.image.ptr == solver._second_whitening_storage[2].ptr
        assert cert.inverse.ptr == branch.transform.ptr
        assert cert.transformed.ptr == branch.metric.v.ptr
        assert (
            cert.work.strides == cert.image.strides == solver._second_whitening_storage[2].strides
        )


@pytest.mark.parametrize("kind", ["nan", "inf", "borderline_rank"])
def test_invalid_padded_whitening_fallback_recovers(kind):
    tol = 1 - 1e-14 if kind == "borderline_rank" else 1e-12
    solver = AllSizePaddedWhitening(Identity(32, wp.get_device()), 2, 4, rank_tol=tol)
    source = np.eye(32)[:8].copy()
    source[::3] = 0
    solver.su.assign(source)
    solver.whiten(solver.su, solver.u1, normalize=True)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver._first_whitening_usage.zero_()
        solver.whiten(solver.su, solver.u1, normalize=True)
    broken = source.copy()
    if kind != "borderline_rank":
        broken[1, 1] = np.nan if kind == "nan" else np.inf
    solver.su.assign(broken)
    wp.capture_launch(capture.graph)
    assert solver._first_whitening_usage.numpy()[1] == 1
    assert solver.status.numpy()[0] == int(kind != "borderline_rank")
    solver.su.assign(source)
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] == 0 and solver.rank.numpy()[0] == 5
    flags = solver.active_u.numpy().astype(bool)
    out = solver.u1.numpy()
    assert np.all(out[~flags] == 0)
    np.testing.assert_allclose(out[flags] @ out[flags].T, np.eye(5), atol=1e-14)
