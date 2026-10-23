"""Diagnostic branch selection must preserve complete uninstrumented solves."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from warp.optim import linear

from benchmarks.experimental.profile_active_block import create_solver, profile


@pytest.mark.parametrize("pool_width", [None, 16])
def test_profile_uploads_comparison_pool_prefix_and_distinguishes_digest(pool_width):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(side=2, modes=6, width=12)
    # None intentionally leaves the new key absent, as in historical configs.
    if pool_width is not None:
        config["initial_pool_width"] = pool_width
    solver, _, _, digest = create_solver(SimpleNamespace(**config))
    expected = np.random.default_rng(42).normal(size=(solver.n, pool_width or 12))[:, :12]
    np.testing.assert_array_equal(solver.initial_vectors.numpy().T, expected)
    if pool_width is not None:
        config.pop("initial_pool_width")
        legacy, _, _, legacy_digest = create_solver(SimpleNamespace(**config))
        assert digest != legacy_digest
        assert not np.array_equal(legacy.initial_vectors.numpy(), solver.initial_vectors.numpy())


def test_selected_branch_profiles_preserve_solver_and_restore_wrappers():
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(side=2, modes=6, width=12)
    solver, check, _, _ = create_solver(SimpleNamespace(**config))
    original_apply = solver.op.apply
    result, _ = profile(solver, check, 300, [0, 1], 2)
    assert solver.op.apply == original_apply
    assert result["before"]["accepted"] and result["after"]["accepted"]
    assert len(result["samples"]) == 4
    for row in result["samples"]:
        assert row["active_rows"] <= row["search_width"] <= solver.b
        assert row["projected_dimension"] == solver.b + row["metric_dimension"]
        assert row["events_ms"]["selected_step"] > 0
        assert row["events_ms"]["cholesky"] > 0
        assert row["events_ms"]["triangular"] > 0
        assert row["events_ms"]["gram"] > 0
        assert all(np.isfinite(v) and v >= 0 for v in row["events_ms"].values())


def test_conditional_second_whitening_is_timed_outside_child_graphs(monkeypatch):
    from benchmarks.experimental import profile_active_block
    from benchmarks.experimental.certified_whitening import CertifiedWhiteningBlock

    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    monkeypatch.setattr(profile_active_block, "BlockLOBPCG", CertifiedWhiteningBlock)
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(side=2, modes=6, width=12)
    solver, check, _, _ = create_solver(SimpleNamespace(**config))
    helpers = [branch._second_whitener for branch in solver._search.buckets]
    original = [helper.apply for helper in helpers]
    original_grams = [branch.gram for branch in solver._search.buckets]
    result, _ = profile(solver, check, 300, [0, 1], 2)
    assert result["before"]["accepted"] and result["after"]["accepted"]
    assert [helper.apply for helper in helpers] == original
    assert [branch.gram for branch in solver._search.buckets] == original_grams
    for sample in result["samples"]:
        assert sample["events_ms"]["second_metric_whitening"] > 0
        assert sample["events_ms"]["metric_eigh"] > 0
        assert all(np.isfinite(t) and t >= 0 for t in sample["events_ms"].values())


@pytest.mark.parametrize("keep_history", [False, True])
def test_installed_first_and_second_whitening_events_stay_outside_conditionals(keep_history):
    import warp as wp

    if not hasattr(linear, "FSAI") or wp.get_device().arch != 89:
        pytest.skip("pinned FSAI and installed Ada certificate required")
    config = json.loads(Path("results/first_whitening/production/100-default-a1.json").read_text())[
        "args"
    ]
    config.update(
        case="clamped_consistent",
        side=3,
        modes=6,
        width=52,
        active_search_keep_history=keep_history,
    )
    solver, check, _, _ = create_solver(SimpleNamespace(**config))
    helpers = [
        helper
        for branch in solver._search.buckets
        for helper in (branch._first_whitener, branch._second_whitener)
        if helper is not None
    ]
    assert helpers
    original = [helper.apply for helper in helpers]
    result, _ = profile(solver, check, 300, [0, 1], 2)
    assert result["before"]["accepted"] and result["after"]["accepted"]
    assert [helper.apply for helper in helpers] == original
    for sample in result["samples"]:
        times = sample["events_ms"]
        assert times["gram"] > 0
        if sample["checkpoint"] == 0:
            assert times["first_metric_whitening"] > 0
            assert times["second_metric_whitening"] > 0
            assert "metric_eigh" not in times
        assert all(np.isfinite(t) and t >= 0 for t in times.values())
        assert "possibly from an earlier step" in sample["metric_sweeps_scope"]


def test_installed_ritz_qr_events_stay_outside_conditional_fallback(monkeypatch):
    import warp as wp

    from benchmarks.experimental import profile_active_block
    from tests.test_installed_ritz_qr import QualifiedBlock

    if not hasattr(linear, "FSAI") or wp.get_device().arch != 89:
        pytest.skip("pinned FSAI and installed Ada reuse required")
    monkeypatch.setattr(profile_active_block, "BlockLOBPCG", QualifiedBlock)
    config = json.loads(Path("results/first_whitening/production/100-default-a1.json").read_text())[
        "args"
    ]
    config.update(case="clamped_consistent", side=5, modes=6, width=104)
    solver, check, _, _ = create_solver(SimpleNamespace(**config))
    helpers = [branch._ritz_qr_reuse for branch in solver._search.buckets]
    assert all(helper is not None for helper in helpers)
    original = [helper.apply for helper in helpers]
    result, _ = profile(solver, check, 300, [0, 1], 2)
    assert result["before"]["accepted"] and result["after"]["accepted"]
    assert [helper.apply for helper in helpers] == original
    for sample in result["samples"]:
        assert sample["events_ms"]["ritz_metric_qr"] > 0
        assert all(np.isfinite(t) and t >= 0 for t in sample["events_ms"].values())
