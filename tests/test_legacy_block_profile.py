"""Optional step profiling must not put event records in conditional children."""

import sys

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.experimental.compare_block import profile_step
from benchmarks.experimental.profile_active_block import capture_profile, checked
from benchmarks.upstream.adapters import matrix
from warpack import BatchedSparseOperator, BlockLOBPCG


def fixture(width, active):
    n, k = 4 * width + 7, 6
    weights = 1.0 + np.arange(n) % 7 / 5.0
    spectrum = np.arange(n) // 2 + 1.0
    operators = [
        BatchedSparseOperator(matrix(sp.diags(x, format="csr")))
        for x in [weights * spectrum, weights, 1 / (weights * (spectrum + 1))]
    ]
    solver = BlockLOBPCG(
        operators[0],
        k,
        width,
        mass=operators[1],
        preconditioner=operators[2],
        tol=1e-10,
        gram_chunk=4096,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
        reuse_operator_images=True,
        search_bucket_size=4 if active else 0,
    )

    def check(values, vectors):
        residual = np.linalg.norm(
            weights[:, None] * (spectrum[:, None] * vectors - vectors * values), axis=0
        ).max()
        orthogonality = np.linalg.norm(vectors.T @ (weights[:, None] * vectors) - np.eye(k))
        accepted = bool(
            residual < 1e-7
            and orthogonality < 1e-9
            and np.max(abs(values - spectrum[:k])) < 1e-9
            and np.linalg.norm(vectors[k:]) < 1e-7
        )
        return dict(accepted=accepted, residual=float(residual), orthogonality=float(orthogonality))

    return solver, check


def wrappers(solver):
    branches = [solver] + ([] if solver._search is None else solver._search.buckets)
    result = [(wp, "launch_tiled"), (solver.preconditioner, "apply")]
    for branch in branches:
        result.extend((branch, name) for name in ["gram", "launch", "rayleigh_ritz", "whiten"])
        result.extend([(branch.op, "apply"), (branch.mass, "apply")])
        for name in ["_ritz_qr_reuse", "_first_whitener", "_second_whitener"]:
            helper = getattr(branch, name, None)
            if helper is not None:
                result.append((helper, "apply"))
    return [(obj, name, getattr(obj, name)) for obj, name in result]


@pytest.mark.parametrize(
    "width,active", [(12, False), (12, True), (52, True), (104, False), (104, True)]
)
def test_static_and_active_steps_preserve_complete_solves_and_wrappers(width, active):
    solver, check = fixture(width, active)
    graph = solver.capture(300)
    wp.capture_launch(graph)
    assert checked(solver, check)["accepted"]
    original = wrappers(solver)
    result = profile_step(solver)
    assert all(getattr(obj, name) == method for obj, name, method in original)
    assert checked(solver, check)["accepted"]
    assert len(result["samples_ms"]) == len(result["selections"]) == 5
    for times, selection in zip(result["samples_ms"], result["selections"], strict=True):
        assert times["selected_step"] > 0 and times["gram"] > 0
        assert times["paired_ritz_rotation"] > 0
        assert times["subtract_projection"] > 0
        assert all(np.isfinite(t) and t >= 0 for t in times.values())
        assert (selection["bucket_index"] is not None) == active
        if width == 104 and wp.get_device().arch == 89:
            assert times["ritz_metric_qr"] > 0
    wp.capture_launch(graph)
    assert checked(solver, check)["accepted"]


def test_capture_failure_restores_all_wrappers(monkeypatch):
    solver, _ = fixture(104, True)
    original = wrappers(solver)

    class RefusedCapture:
        def __enter__(self):
            raise RuntimeError("capture refused")

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(wp, "ScopedCapture", lambda **kwargs: RefusedCapture())
    with pytest.raises(RuntimeError, match="capture refused"):
        capture_profile(solver, 0)
    assert all(getattr(obj, name) == method for obj, name, method in original)


@pytest.mark.parametrize("active", [False, True])
def test_branch_index_contract_precedes_instrumentation(active):
    solver, _ = fixture(12, active)
    original = wrappers(solver)
    with pytest.raises(ValueError, match="branch index"):
        capture_profile(solver, None if active else 0)
    assert all(getattr(obj, name) == method for obj, name, method in original)


@pytest.mark.parametrize("reuse", [False, True])
def test_cli_profile_allows_active_search_only_with_image_reuse(monkeypatch, tmp_path, reuse):
    from benchmarks.experimental import compare_block

    class Prepared(Exception):
        pass

    def prepare(args, **kwargs):
        assert args.profile and args.active_search_bucket_size == 4 and args.reuse_x_image
        raise Prepared()

    monkeypatch.setattr(compare_block, "prepare", prepare)
    monkeypatch.setattr(compare_block, "other_gpu_processes", lambda: [])
    args = [
        "compare",
        "--profile",
        "--active-search-bucket-size",
        "4",
        "--out",
        str(tmp_path / "result.json"),
    ]
    if reuse:
        args.append("--reuse-x-image")
    monkeypatch.setattr(sys, "argv", args)
    if reuse:
        with pytest.raises(Prepared):
            compare_block.main()
    else:
        with pytest.raises(SystemExit) as caught:
            compare_block.main()
        assert caught.value.code == 2
