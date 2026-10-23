"""Diagnostic branch selection must preserve complete uninstrumented solves."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from warp.optim import linear

from benchmarks.experimental.profile_active_block import create_solver, profile


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
        assert all(np.isfinite(v) and v >= 0 for v in row["events_ms"].values())
