"""Installed odd-size search views and independent certificate bounds."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from tests.test_first_whitening import AllSizeInstalledWhitening
from warpack import BlockLOBPCG
from warpack.block import Identity


@pytest.mark.parametrize("compact", [False, True])
def test_odd_search_branch_padding_and_dependent_fallback(compact):
    if wp.get_device().arch != 89:
        pytest.skip("installed specialization qualified on Ada")
    solver = BlockLOBPCG(
        Identity(180, wp.get_device()),
        4,
        51,
        reuse_operator_images=True,
        search_bucket_size=4,
        keep_history=True,
        compact_projected_graphs=compact,
    )
    branches = [b for b in solver._search.buckets if b.su.shape[0] % 2 and b._first_whitener]
    assert branches
    for branch in branches:
        m = branch.su.shape[0]
        source = la.qr(np.random.default_rng(m).normal(size=(solver.n, m)), mode="economic")[
            0
        ].T.copy()
        branch.su.assign(source)
        branch.whiten(branch.su, branch.u1, normalize=True)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            branch.status.zero_()
            solver._first_whitening_usage.zero_()
            branch.whiten(branch.su, branch.u1, normalize=True)
            branch.whiten(branch.u1, branch.u2)
        for kind in ["full", "padding", "dependent", "full"]:
            current = source.copy()
            if kind == "padding":
                current[::3] = 0
            if kind == "dependent":
                current[1] = current[0]
            branch.su.assign(current)
            wp.capture_launch(capture.graph)
            expected_rank = m - (
                len(current[::3]) if kind == "padding" else int(kind == "dependent")
            )
            assert branch.status.numpy()[0] == 0 and branch.rank.numpy()[0] == expected_rank
            usage = solver._first_whitening_usage.numpy()
            np.testing.assert_array_equal(
                usage[:2], [int(kind != "dependent"), int(kind == "dependent")]
            )
            flags = branch.active_u.numpy().astype(bool)
            result = branch.u2.numpy()
            assert np.all(result[~flags] == 0)
            selected = result[flags]
            assert la.norm(selected @ selected.T - np.eye(expected_rank)) < 1e-10
            _, _, vh = la.svd(current, full_matrices=False)
            wanted = vh[:expected_rank]
            assert la.norm(wanted - (wanted @ selected.T) @ selected) < 1e-9


@pytest.mark.parametrize("condition", [1.0, 1e4, 1e8, 1e12, 1e14])
def test_installed_bounds_against_extended_precision(condition):
    n, m, rank = 40, 24, 13
    solver = AllSizeInstalledWhitening(Identity(n, wp.get_device()), 4, m // 2)
    helper = solver._first_whitener
    solver.su.assign(np.eye(n)[:m])
    solver.whiten(solver.su, solver.u1, normalize=True)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver.whiten(solver.su, solver.u1, normalize=True)
    accepted_count = 0
    for seed in range(4):
        rng = np.random.default_rng(3171 + seed)
        q = la.qr(rng.normal(size=(rank, rank)))[0]
        active = q * np.sqrt(np.geomspace(1 / condition, 1, rank))
        source = np.zeros((m, n))
        indices = rng.permutation(m)[:rank]
        source[indices, :rank] = active
        solver.su.assign(source)
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == 0
        if helper.accepted.numpy()[0]:
            accepted_count += 1
            # The certificate is about the computed normalized Gram, not an
            # assertion that formation of that Gram had zero roundoff.
            gram = solver.metric.g.numpy()
            spectrum = la.eigvalsh(gram[np.ix_(indices, indices)])
            delta, lower, upper, _ = helper.diagnostic.numpy()
            assert lower > solver.rank_tol * upper
            assert lower <= spectrum[0] + 1e-14 and upper >= spectrum[-1]
            inverse = helper.inverse.numpy().astype(np.longdouble)
            auxiliary = helper.matrix.numpy().astype(np.longdouble)
            error = inverse @ auxiliary @ inverse.T - np.eye(m, dtype=np.longdouble)
            assert np.max(np.sum(abs(error), axis=1)) <= delta <= 0.25
            assert spectrum[0] > solver.rank_tol * spectrum[-1]
            assert solver.rank.numpy()[0] == rank
    if condition == 1.0:
        assert accepted_count == 4


def test_largest_qualified_metric_rank_and_replay(tmp_path):
    from tests.test_first_whitening import test_first_pass_rank_scale_and_replay as check

    check(256, tmp_path, AllSizeInstalledWhitening)
