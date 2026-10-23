"""Installed first-whitening guards, shared views and device fallback."""

import numpy as np
import pytest
import warp as wp

from tests.test_first_whitening import AllSizeInstalledWhitening, failed_factor
from warpack import BlockLOBPCG
from warpack.block import Identity


class CanonicalBlock(BlockLOBPCG):
    _first_whitening_enabled = False


@pytest.mark.parametrize("width", [47, 48, 256, 257])
def test_default_guard(width):
    solver = BlockLOBPCG(Identity(3 * width + 1, wp.get_device()), 4, width)
    eligible = wp.get_device().arch == 89 and 96 <= 2 * width <= 512
    assert (solver._first_whitener is not None) == eligible
    if eligible:
        assert solver._first_whitening_storage is solver._second_whitening_storage
    else:
        assert solver._first_whitening_storage is None
        assert solver._first_whitening_usage is None


@pytest.mark.parametrize("compact", [False, True])
def test_shared_branch_views_padding_and_sentinels(compact):
    if wp.get_device().arch != 89:
        pytest.skip("installed specialization qualified on Ada")
    solver = BlockLOBPCG(
        Identity(400, wp.get_device()),
        4,
        52,
        reuse_operator_images=True,
        search_bucket_size=4,
        compact_projected_graphs=compact,
    )
    storage = solver._first_whitening_storage
    assert storage is solver._second_whitening_storage and len(storage) == 3
    for branch in [solver, *solver._search.buckets]:
        helper = branch._first_whitener
        m = branch.su.shape[0]
        assert (helper is not None) == (96 <= m <= 512)
        if helper is None:
            continue
        assert helper.inverse.ptr == branch.transform.ptr
        assert helper.transformed.ptr == branch.metric.v.ptr
        for array, backing in zip(
            (helper.matrix, helper.factor, helper.work), storage, strict=True
        ):
            assert array.ptr == backing.ptr and array.strides == backing.strides
        source = np.eye(solver.n)[:m].copy()
        source[::3] = 0
        branch.su.assign(source)
        branch.whiten(branch.su, branch.u1, normalize=True)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            branch.status.zero_()
            branch.whiten(branch.su, branch.u1, normalize=True)
        for backing in storage:
            backing.fill_(17.0)
        branch.su.assign(source)
        wp.capture_launch(capture.graph)
        assert branch.status.numpy()[0] == 0
        assert branch.rank.numpy()[0] == np.count_nonzero(source.sum(axis=1))
        np.testing.assert_array_equal(branch.u1.numpy(), source)
        np.testing.assert_array_equal(branch.active_u.numpy(), source.sum(axis=1))
        assert helper.accepted.numpy()[0] == 1
        for backing in storage:
            values = backing.numpy()
            assert np.all(values[m:, :] == 17.0) and np.all(values[:, m:] == 17.0)


@pytest.mark.parametrize("padded", [False, True])
def test_failed_installed_factor_is_bitwise_canonical(monkeypatch, padded):
    from warpack import _first_whitening

    monkeypatch.setattr(_first_whitening, "global_cholesky", failed_factor)
    source = np.random.default_rng(2171).normal(size=(8, 32))
    if padded:
        source[::3] = 0
    results = []
    for solver_type in [CanonicalBlock, AllSizeInstalledWhitening]:
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
        assert solver.rank.numpy()[0] == (5 if padded else 8)
        results.append(solver.u1.numpy())
        if solver_type is AllSizeInstalledWhitening:
            np.testing.assert_array_equal(solver._first_whitening_usage.numpy(), [0, 1, 0])
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("kind", ["nan", "inf", "borderline_rank"])
def test_uncertified_input_recovers(kind):
    rank_tol = 1 - 1e-14 if kind == "borderline_rank" else 1e-12
    solver = AllSizeInstalledWhitening(Identity(32, wp.get_device()), 2, 4, rank_tol=rank_tol)
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
    output = solver.u1.numpy()
    assert np.all(output[~flags] == 0)
    np.testing.assert_allclose(output[flags] @ output[flags].T, np.eye(5), atol=1e-14)
