"""Second-pass certificates must preserve rank and select canonical fallback."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.certified_whitening import CertifiedWhiteningBlock
from warpack.block import Identity


@pytest.mark.parametrize("width", [4, 24, 52, 104])
def test_rank_changes_certificate_and_canonical_replay(width):
    n, m = 3 * width + 7, 2 * width
    solver = CertifiedWhiteningBlock(Identity(n, wp.get_device()), min(width, 4), width)
    rng = np.random.default_rng(162)
    orthogonal = la.qr(rng.normal(size=(n, m)), mode="economic")[0].T.copy()
    solver.u1.assign(orthogonal)
    solver.whiten(solver.u1, solver.u2)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.whitening_usage.zero_()
        solver.whiten(solver.u1, solver.u2)
    for kind in ["full", "partial", "dependent", "scaled", "inactive_mismatch", "zero", "full"]:
        source, active = orthogonal.copy(), np.ones(m, dtype=np.int32)
        fast, rank = True, m
        if kind == "partial":
            source[-3:] = 0
            active[-3:] = 0
            rank = m - 3
        elif kind == "dependent":
            source[1] = source[0] + 1e-8 * source[1]
            fast, rank = False, m - 1
        elif kind == "scaled":
            source[0] *= 0.1
            fast = False
        elif kind == "inactive_mismatch":
            active[-1] = 0
            fast = False
        elif kind == "zero":
            source[:] = 0
            active[:] = 0
            rank = 0
        solver.u1.assign(source)
        solver.active_u.assign(active)
        solver.status.zero_()
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == 0
        assert solver.rank.numpy()[0] == rank
        np.testing.assert_array_equal(solver.whitening_usage.numpy(), [int(fast), int(not fast)])
        output, flags = solver.u2.numpy(), solver.active_u.numpy().astype(bool)
        assert np.count_nonzero(flags) == rank
        assert np.max(abs(output[~flags]), initial=0) == 0
        if rank:
            selected = output[flags]
            assert np.linalg.norm(selected @ selected.T - np.eye(rank)) < 1e-10
            # Independent SVD checks the full retained row-space, not vector signs.
            _, _, vh = la.svd(source, full_matrices=False)
            assert np.linalg.norm(vh[:rank] - (vh[:rank] @ selected.T) @ selected) < 1e-7


@wp.kernel
def failed_factor(
    matrix: wp.array2d[wp.float64],
    factor: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    work: wp.array2d[wp.float64],
):
    status[0] = 1


def test_factor_failure_uses_unmodified_canonical_matrix(monkeypatch):
    from benchmarks.experimental import certified_whitening

    monkeypatch.setattr(certified_whitening, "global_cholesky", failed_factor)
    solver = CertifiedWhiteningBlock(Identity(32, wp.get_device()), 2, 4)
    source = np.eye(32)[:8].copy()
    solver.u1.assign(source)
    solver.whiten(solver.u1, solver.u2)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.whitening_usage.zero_()
        solver.whiten(solver.u1, solver.u2)
    solver.active_u.fill_(1)
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] == 0
    assert solver.rank.numpy()[0] == 8
    np.testing.assert_array_equal(solver.whitening_usage.numpy(), [0, 1])
    np.testing.assert_array_equal(solver.u2.numpy(), source)


def test_rank_threshold_certificate_and_nonfinite_recovery():
    solver = CertifiedWhiteningBlock(Identity(32, wp.get_device()), 1, 4, rank_tol=0.9)
    source = np.eye(32)[:8].copy()
    solver.u1.assign(source)
    solver.whiten(solver.u1, solver.u2)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.whitening_usage.zero_()
        solver.whiten(solver.u1, solver.u2)
    for kind in ["rank_threshold", "nonfinite", "recover"]:
        x = source.copy()
        if kind == "rank_threshold":
            x[0] *= np.sqrt(0.8)
            x[-1] *= np.sqrt(1.2)
        elif kind == "nonfinite":
            x[0, 0] = np.nan
        solver.u1.assign(x)
        solver.active_u.fill_(1)
        solver.status.zero_()
        wp.capture_launch(capture.graph)
        if kind == "nonfinite":
            assert solver.status.numpy()[0] != 0
        else:
            assert solver.status.numpy()[0] == 0
            assert solver.rank.numpy()[0] == (1 if kind == "rank_threshold" else 8)
        np.testing.assert_array_equal(
            solver.whitening_usage.numpy(), [int(kind == "recover"), int(kind != "recover")]
        )


@pytest.mark.parametrize("case", ["clamped_consistent", "free", "heterogeneous"])
def test_complete_fem_spectrum_with_certified_second_pass(case, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from warp.optim import linear

    from benchmarks.experimental import profile_active_block

    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    monkeypatch.setattr(profile_active_block, "BlockLOBPCG", CertifiedWhiteningBlock)
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(case=case, side=3, modes=6, width=24)
    solver, check, _, _ = profile_active_block.create_solver(SimpleNamespace(**config))
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        quality = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert quality["accepted"], quality
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == solver.k
        assert solver.whitening_usage.numpy()[0] > 0


def test_coupled_mass_constraints_clusters_and_changed_problem(tmp_path):
    import scipy.sparse as sp

    from benchmarks.upstream.adapters import matrix
    from warpack import SparseOperator

    n, k, width = 80, 6, 16

    def fixture(seed):
        rng = np.random.default_rng(seed)
        lower = np.eye(n) + np.diag(np.linspace(0.05, 0.25 + seed * 0.001, n - 1), -1)
        mass = lower @ lower.T
        q = la.qr(rng.normal(size=(n, n)))[0]
        modes = la.solve_triangular(lower.T, q, lower=False)
        values = np.repeat(np.arange(n // 2, dtype=float), 2)
        a = mass @ (modes * values) @ modes.T @ mass
        constraints = (modes[:, :2] @ np.array([[2.0, 0.4], [0.0, 3.0]])).T.copy()
        return a, mass, modes, values, constraints

    a, mass, modes, values, constraints = fixture(19)
    op, metric, pre = [
        SparseOperator(matrix(sp.csr_matrix(x))) for x in (a, mass, la.inv(a + mass))
    ]
    basis = wp.array(constraints, dtype=wp.float64)
    solver = CertifiedWhiteningBlock(
        op,
        k,
        width=width,
        mass=metric,
        preconditioner=pre,
        constraints=basis,
        normalize_constraints=True,
        tol=1e-10,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
        reuse_operator_images=True,
        search_bucket_size=4,
        keep_history=True,
    )
    graph = solver.capture(200)
    for seed in [19, 31, 19]:
        a, mass, modes, values, constraints = fixture(seed)
        for target, host in [(op, a), (metric, mass), (pre, la.inv(a + mass))]:
            csr = sp.csr_matrix(host)
            np.testing.assert_array_equal(target.matrix.columns.numpy(), csr.indices)
            target.matrix.values.assign(csr.data)
        basis.assign(constraints)
        wp.capture_launch(graph)
        result, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        assert solver.whitening_usage.numpy()[0] > 0
        np.testing.assert_allclose(result, values[2 : 2 + k], atol=1e-10)
        assert np.linalg.norm(vectors.T @ mass @ vectors - np.eye(k)) < 1e-10
        assert np.linalg.norm(a @ vectors - (mass @ vectors) * result) < 1e-8
        wanted = modes[:, 2 : 2 + k]
        assert np.linalg.norm(vectors - wanted @ (wanted.T @ mass @ vectors)) < 1e-8
        assert np.linalg.norm(constraints @ mass @ vectors) < 1e-9
        np.testing.assert_array_equal(basis.numpy(), constraints)
    dot = tmp_path / "certified.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    content = dot.read_text()
    assert "HtoD" not in content and "DtoH" not in content and "HOST" not in content
