"""Experimental first whitening must preserve rank, fallback and physical modes."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.compare_block import prepare
from benchmarks.experimental.first_whitening import FirstWhiteningBlock
from benchmarks.experimental.padded_first_whitening import PaddedFirstWhiteningBlock
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG
from warpack.block import Identity


class AllSizeInstalledWhitening(BlockLOBPCG):
    _first_whitening_minimum = 2
    _first_whitening_arches = None


class AllSizeFirstWhitening(FirstWhiteningBlock):
    _first_metric_minimum = 2


class AllSizePaddedWhitening(PaddedFirstWhiteningBlock):
    _first_metric_minimum = 2


@pytest.mark.parametrize(
    "solver_type", [AllSizeFirstWhitening, AllSizePaddedWhitening, AllSizeInstalledWhitening]
)
@pytest.mark.parametrize("width", [4, 24, 52, 104, 204])
def test_first_pass_rank_scale_and_replay(width, tmp_path, solver_type):
    n, m = 3 * width + 7, 2 * width
    solver = solver_type(Identity(n, wp.get_device()), min(width, 4), width)
    basis = la.qr(np.random.default_rng(171).normal(size=(n, m)), mode="economic")[0].T.copy()
    solver.su.assign(basis)
    solver.whiten(solver.su, solver.u1, normalize=True)
    solver.whiten(solver.u1, solver.u2)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver._first_whitening_usage.zero_()
        solver.whiten(solver.su, solver.u1, normalize=True)
        solver.whiten(solver.u1, solver.u2)
    for kind in ["full", "scaled", "partial", "dependent", "zero", "full"]:
        source, expected_rank, fast = basis.copy(), m, True
        if kind == "scaled":
            source *= np.geomspace(1e-80, 1e80, m)[:, None]
        elif kind == "partial":
            source[-3:] = 0
            expected_rank, fast = m - 3, solver_type is not AllSizeFirstWhitening
        elif kind == "dependent":
            source[1] = source[0] + 1e-8 * source[1]
            expected_rank, fast = m - 1, False
        elif kind == "zero":
            source[:] = 0
            expected_rank, fast = 0, False
        solver.su.assign(source)
        wp.capture_launch(capture.graph)
        assert solver.status.numpy()[0] == 0
        assert solver.rank.numpy()[0] == expected_rank
        usage = solver._first_whitening_usage.numpy()
        np.testing.assert_array_equal(usage[:2], [int(fast), int(not fast)])
        assert usage[2] == int(
            kind == "zero" or (kind == "partial" and solver_type is AllSizeFirstWhitening)
        )
        flags, output = solver.active_u.numpy().astype(bool), solver.u2.numpy()
        assert np.count_nonzero(flags) == expected_rank
        assert np.max(abs(output[~flags]), initial=0) == 0
        if expected_rank:
            selected = output[flags]
            assert la.norm(selected @ selected.T - np.eye(expected_rank)) < 1e-10
            norms = la.norm(source, axis=1)
            normalized = source / np.where(norms > 0, norms, 1)[:, None]
            _, _, vt = la.svd(normalized, full_matrices=False)
            assert la.norm(vt[:expected_rank] - (vt[:expected_rank] @ selected.T) @ selected) < 1e-7
    path = tmp_path / "first-whitening.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    graph = graph_summary(path)
    assert not graph["node_types_including_conditional_body"].get("HOST")
    assert all("DtoD" in kind for kind in graph["memcpy_kinds"])


@wp.kernel(enable_backward=False)
def failed_factor(
    g: wp.array2d[wp.float64],
    factor: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    work: wp.array2d[wp.float64],
):
    status[0] = 1


def test_failed_attempt_keeps_original_gram_and_outer_status(monkeypatch):
    from benchmarks.experimental import first_metric_certificate

    monkeypatch.setattr(first_metric_certificate, "global_cholesky", failed_factor)
    source = np.random.default_rng(2171).normal(size=(8, 32))
    results = []
    for solver_type in [BlockLOBPCG, AllSizeFirstWhitening]:
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
        assert solver.rank.numpy()[0] == 8
        results.append(solver.u1.numpy())
        if solver_type is AllSizeFirstWhitening:
            np.testing.assert_array_equal(solver._first_whitening_usage.numpy(), [0, 1, 0])
    np.testing.assert_array_equal(results[0], results[1])


def test_first_and_second_share_scratch_across_search_branches():
    solver = FirstWhiteningBlock(
        Identity(400, wp.get_device()), 4, 52, reuse_operator_images=True, search_bucket_size=4
    )
    assert solver._first_whitening_storage is solver._second_whitening_storage
    for branch in [solver, *solver._search.buckets]:
        if branch._first_whitener is None:
            continue
        certificate = branch._first_whitener.certificate
        assert certificate.inverse.ptr == branch.transform.ptr
        assert certificate.transformed.ptr == branch.metric.v.ptr
        assert certificate.factor.ptr == solver._second_whitening_storage[1].ptr
        assert certificate.work.ptr == solver._second_whitening_storage[2].ptr
        assert certificate.image.ptr == solver._second_whitening_storage[0].ptr


@pytest.mark.parametrize(
    "solver_type", [FirstWhiteningBlock, PaddedFirstWhiteningBlock, BlockLOBPCG]
)
@pytest.mark.parametrize(
    "case", ["clamped_consistent", "clamped_lumped", "free", "disconnected", "heterogeneous"]
)
@pytest.mark.parametrize("keep_history", [True, False])
def test_fem_modes_and_nullspace_with_first_certificate(case, keep_history, solver_type):
    if not (hasattr(linear, "FSAI") and hasattr(linear.LinearOperator, "batch_offsets")):
        pytest.skip("pinned upstream FSAI and batched CG required")
    args = SimpleNamespace(case=case, side=2, modes=30, shift=0.01)
    metadata = {}
    a, mass, shifted, constraints, _, _, k, check = prepare(args, metadata=metadata)
    width = 52
    factor = linear.FSAI(shifted, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
    cg = BatchedCGPreconditioner(
        shifted,
        width,
        tol=0.01,
        maxiter=25,
        preconditioner=factor,
        working_dtype=wp.float32,
        fsai_layout="columns",
        matrix_layout="columns",
    )
    solver = solver_type(
        BatchedSparseOperator(a),
        k,
        width,
        mass=BatchedSparseOperator(mass),
        constraints=constraints,
        preconditioner=cg,
        tol=1e-11,
        initial_vectors=wp.array(
            np.random.default_rng(42).normal(size=(width, a.shape[0])), dtype=wp.float64
        ),
        gram_chunk=4096,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
        reuse_operator_images=True,
        search_bucket_size=4,
        keep_history=keep_history,
        preconditioner_bucket_size=4,
    )
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        quality = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert solver.status.numpy()[0] == 0, quality
        assert solver.converged.numpy()[0] == k, quality
        assert quality["accepted"], quality
        assert quality["recovered_nullity"] == metadata["retained_rigid_modes"]
        assert solver._first_whitening_usage.numpy()[0] > 0


@pytest.mark.parametrize("kind", ["nan", "inf", "borderline_rank"])
def test_invalid_or_borderline_metric_uses_fallback_then_recovers(kind):
    rank_tol = 1.0 - 1e-14 if kind == "borderline_rank" else 1e-12
    solver = AllSizeFirstWhitening(Identity(32, wp.get_device()), 2, 4, rank_tol=rank_tol)
    source = np.eye(32)[:8].copy()
    solver.su.assign(source)
    solver.whiten(solver.su, solver.u1, normalize=True)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.status.zero_()
        solver._first_whitening_usage.zero_()
        solver.whiten(solver.su, solver.u1, normalize=True)
    broken = source.copy()
    if kind != "borderline_rank":
        broken[0, 0] = np.nan if kind == "nan" else np.inf
    solver.su.assign(broken)
    wp.capture_launch(capture.graph)
    assert solver._first_whitening_usage.numpy()[1] == 1
    assert solver.status.numpy()[0] == int(kind != "borderline_rank")
    solver.su.assign(source)
    wp.capture_launch(capture.graph)
    assert solver.status.numpy()[0] == 0
    assert solver.rank.numpy()[0] == 8
    np.testing.assert_array_equal(solver.u1.numpy(), source)
