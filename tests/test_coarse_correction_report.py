"""A correction experiment cannot hide coarse cost, bad spectra or host work."""

import copy

import pytest

from benchmarks.applications.fem_coarse_correction_report import compare, summarize


def fixture():
    q = dict(
        accepted=True,
        status=0,
        converged=20,
        residual=1e-12,
        orthogonality=1e-14,
        spectrum_error=1e-12,
        requested_subspace_error=1e-10,
        iterations=24,
    )
    row = dict(
        accepted=True,
        correction_status=0,
        quality=q,
        source_quality=q | dict(converged=24, residual=1e-4),
        source_seconds=0.1,
        fine_seconds=0.2,
        total_seconds=0.3,
    )
    setup = dict.fromkeys(
        (
            "fine_assembly_seconds",
            "fine_setup_seconds",
            "fine_capture_seconds",
            "coarse_assembly_seconds",
            "coarse_setup_capture_seconds",
            "coarse_initial_solve_seconds",
            "transfer_setup_seconds",
        ),
        1.0,
    )
    raw = dict(
        accepted=True,
        arguments=dict(profile="balanced-inner-cg", repeats=3, modes=20, width=24, coarse_tol=1e-3),
        warmup=copy.deepcopy(row),
        rows=[copy.deepcopy(row) for _ in range(3)],
        source_count=24,
        source_tolerance=1e-3,
        setup=setup,
        setup_total_seconds=7.0,
        setup_capture_device_free_memory_decrease_bytes=1024,
        graphs={
            level: dict(node_types_including_conditional_body={"KERNEL": 4}, memcpy_kinds={})
            for level in ("fine", "coarse")
        },
        matrix_sha256={},
        source_sha256={},
        dependency={},
        environment={},
        n=34560,
    )
    return raw, dict(timing_eligible=True, exit_code=0)


def test_accounting_and_performance_regression():
    raw, occupancy = fixture()
    summary = summarize(raw, occupancy)
    assert summary["first_use_component_seconds"] == 7.3
    assert summary["total_seconds"] == 0.3
    assert compare(summary, summary) == 1
    slower = copy.deepcopy(summary)
    slower["total_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(slower, summary)
    changed = copy.deepcopy(summary)
    changed["signature"]["matrix_sha256"]["fine"] = "changed"
    with pytest.raises(ValueError, match="incompatible"):
        compare(changed, summary)
    with pytest.raises(ValueError, match="manually excluded"):
        summarize(raw, occupancy, dict(reason="known CPU contention"))


@pytest.mark.parametrize("profile", ["geometric", "geometric-inner-cg"])
def test_geometric_hierarchy_charges_setup_without_source_eigensolve(profile):
    raw, occupancy = fixture()
    raw["arguments"]["profile"] = profile
    raw["graphs"].pop("coarse")
    raw["source_count"] = raw["source_tolerance"] = None
    raw["setup"] = {k: v for k, v in raw["setup"].items() if k.startswith("fine_")}
    raw["setup_total_seconds"] = 3.0
    for row in [raw["warmup"], *raw["rows"]]:
        row.update(source_quality=None, source_seconds=0.0, total_seconds=0.2)
    assert summarize(raw, occupancy)["first_use_component_seconds"] == 3.2
    raw["rows"][1]["source_seconds"] = 0.1
    with pytest.raises(ValueError, match="unexpected coarse source"):
        summarize(raw, occupancy)


@pytest.mark.parametrize(
    "failure",
    [
        "contended",
        "partial",
        "native",
        "spectrum",
        "nan",
        "coarse",
        "rank",
        "host",
        "missing-audit",
        "setup",
        "hidden-source-cost",
        "wrong-source-count",
    ],
)
def test_failed_run_rejected(failure):
    raw, occupancy = fixture()
    row = raw["rows"][1]
    if failure == "contended":
        occupancy["timing_eligible"] = False
    elif failure == "partial":
        raw["rows"].pop()
    elif failure == "native":
        row["quality"]["converged"] = 19
    elif failure == "spectrum":
        row["quality"]["spectrum_error"] = 0.142
    elif failure == "nan":
        row["fine_seconds"] = float("nan")
    elif failure == "coarse":
        row["source_quality"]["status"] = 1
    elif failure == "rank":
        row["correction_status"] = 1
    elif failure == "host":
        raw["graphs"]["fine"]["memcpy_kinds"]["DtoH"] = 1
    elif failure == "missing-audit":
        del raw["graphs"]["coarse"]
    elif failure == "setup":
        del raw["setup"]["coarse_initial_solve_seconds"]
    elif failure == "hidden-source-cost":
        row["total_seconds"] = row["fine_seconds"]
    else:
        raw["source_count"] = 20
    with pytest.raises(ValueError):
        summarize(raw, occupancy)
