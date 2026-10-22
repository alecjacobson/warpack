"""A failed or incomplete hierarchy run must never become a speedup claim."""

import copy

import pytest

from benchmarks.applications.fem_hierarchy_report import compare, summarize


def fixture():
    q = dict(
        accepted=True,
        status=0,
        converged=20,
        residual=1e-12,
        orthogonality=1e-14,
        spectrum_error=1e-12,
        requested_subspace_error=1e-10,
        iterations=30,
    )
    seq = [
        dict(
            strategy=s,
            coarse_seconds=0.1 if s == "coarse" else 0,
            transfer_seconds=0,
            solve_seconds=1,
            total_seconds=1.1 if s == "coarse" else 1,
            quality=q.copy(),
            coarse_quality=q.copy() if s == "coarse" else None,
        )
        for s in ("random", "coarse", "retained")
    ]
    r = dict(
        accepted=True,
        arguments=dict(coarse=5, fine=15, modes=20, width=24, mass="consistent", repeats=3),
        warmup=copy.deepcopy(seq),
        rows=[copy.deepcopy(seq) for _ in range(3)],
        setup={
            name: 1.0
            for name in (
                "fine_assembly_seconds",
                "fine_setup_capture_seconds",
                "coarse_assembly_seconds",
                "coarse_setup_capture_seconds",
                "transfer_setup_capture_seconds",
            )
        },
        graphs={
            level: dict(node_types_including_conditional_body={"KERNEL": 1}, memcpy_kinds={})
            for level in ("fine", "coarse", "transfer_refine")
        },
        source_sha256={},
        matrix_sha256={},
        warp="test",
        versions={},
        dimensions={},
    )
    return r, dict(timing_eligible=True, exit_code=0)


def test_complete_run_component_costs():
    result = summarize(*fixture())
    assert result["rows"][0]["first_use_component_seconds"] == 3
    assert result["rows"][1]["first_use_component_seconds"] == 6.1
    assert result["rows"][2]["first_use_component_seconds"] is None


@pytest.mark.parametrize(
    "mutation", ["contended", "partial", "native", "nan", "wrong-spectrum", "coarse", "host"]
)
def test_bad_run_rejected(mutation):
    raw, occupancy = fixture()
    row = raw["rows"][1][1]
    if mutation == "contended":
        occupancy["timing_eligible"] = False
    elif mutation == "partial":
        raw["rows"].pop()
    elif mutation == "native":
        row["quality"]["status"] = 1  # Aggregate accepted flag is deliberately stale.
    elif mutation == "nan":
        row["solve_seconds"] = float("nan")
    elif mutation == "wrong-spectrum":
        row["quality"]["spectrum_error"] = 0.142
    elif mutation == "host":
        raw["graphs"]["transfer_refine"]["memcpy_kinds"]["DtoH"] = 1
    else:
        row["coarse_quality"] = None
    with pytest.raises(ValueError):
        summarize(raw, occupancy)


def test_workflow_performance_regression_gate():
    baseline = summarize(*fixture()) | {"environment": {"gpu": "same"}}
    current = copy.deepcopy(baseline)
    assert set(compare(current, baseline).values()) == {1.0}
    current["rows"][1]["total_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(current, baseline)
    current = copy.deepcopy(baseline)
    current["environment"]["gpu"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        compare(current, baseline)
