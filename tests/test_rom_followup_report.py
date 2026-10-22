"""Rejected physical answers and contaminated timings never become comparisons."""

import copy

import pytest

from benchmarks.applications.rom_followup_report import compare, summarize


def fixture(kind):
    q = dict(
        accepted=True,
        status=0,
        converged=4,
        positive=4,
        cg_stats=[32, 8, 0, 4],
        residual=1e-12,
        orthogonality=1e-14,
        spectrum_error=1e-12,
        requested_subspace_error=1e-11,
        source_residual=1e-13,
        subspace_error=1e-11,
        energy_error=1e-12,
    )
    args = dict(modes=4, repeats=3, out="x.json", backend="cg", profile="full-loose", width=8)
    row = dict(
        quality=q,
        accepted=True,
        source_seconds=0.1,
        fine_seconds=0.2,
        response_seconds=0.1,
        modes_seconds=0.2,
        total_seconds=0.3,
        source_quality=dict(status=0, converged=8, residual=1e-4, orthogonality=1e-14),
    )
    raw = dict(
        dependency=dict(
            warp_version="test", files={"linear": {"sha256": "same"}}, manifest_status="absent"
        ),
        accepted=True,
        arguments=args,
        setup={"assembly": 1, "solver": 2},
        assembly_seconds=1,
        setup_seconds=1,
        capture_seconds=1,
        source_tolerance=1e-3,
        source_count=8,
        warmup=copy.deepcopy(row),
        rows=[copy.deepcopy(row) for _ in range(3)],
        graphs={
            name: dict(node_types_including_conditional_body={"KERNEL": 1}, memcpy_kinds={})
            for name in (("fine", "coarse") if kind == "hierarchy" else ("response", "modes"))
        },
        source_sha256="source",
        matrix_sha256="matrix",
        forces_sha256="forces",
    )
    return raw, dict(timing_eligible=True, exit_code=0)


@pytest.mark.parametrize("kind", ["hierarchy", "forces"])
def test_complete_accounting_and_performance_gate(kind):
    raw, occ = fixture(kind)
    r = summarize(raw, occ, kind) | {"environment": {"gpu": "same"}}
    assert r["first_use_seconds"] == 3.3
    assert compare(r, r) == 1
    slower = copy.deepcopy(r)
    slower["total_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(slower, r)


@pytest.mark.parametrize("kind", ["hierarchy", "forces"])
@pytest.mark.parametrize(
    "failure", ["contended", "nan", "native", "subspace", "partial", "copy", "source", "graph"]
)
def test_corrupt_trial_rejected(kind, failure):
    raw, occ = fixture(kind)
    row = raw["rows"][1]
    if failure == "contended":
        occ["timing_eligible"] = False
    elif failure == "nan":
        row["quality"]["residual"] = float("nan")
    elif failure == "native":
        row["quality"]["status"] = 1
    elif failure == "subspace":
        row["quality"]["subspace_error"] = row["quality"]["requested_subspace_error"] = 1e-3
    elif failure == "partial":
        raw["rows"].pop()
    elif failure == "copy":
        next(iter(raw["graphs"].values()))["memcpy_kinds"]["DtoH"] = 1
    elif failure == "graph":
        raw["graphs"].pop(next(iter(raw["graphs"])))
    elif kind == "hierarchy":
        row["source_quality"]["converged"] = 7
    else:
        row["quality"]["positive"] = 3
    with pytest.raises(ValueError):
        summarize(raw, occ, kind)


def test_force_identity_required_for_timing_comparison():
    raw, occ = fixture("forces")
    baseline = summarize(raw, occ, "forces") | {"environment": {"gpu": "same"}}
    raw["forces_sha256"] = "different forces"
    changed = summarize(raw, occ, "forces") | {"environment": baseline["environment"]}
    with pytest.raises(ValueError, match="forces_sha256"):
        compare(changed, baseline)


def test_helper_change_rejects_identical_driver_configuration():
    raw, occ = fixture("forces")
    raw["source_sha256"] = {"driver.py": "same", "reference.py": "old"}
    baseline = copy.deepcopy(summarize(raw, occ, "forces")) | {"environment": {"gpu": "same"}}
    raw["source_sha256"]["reference.py"] = "changed"
    changed = summarize(raw, occ, "forces") | {"environment": baseline["environment"]}
    with pytest.raises(ValueError, match="source_sha256"):
        compare(changed, baseline)


@pytest.mark.parametrize("failure", [None, "missing-lift", "host", "copy", "mass", "rank"])
def test_weighted_svd_stage_contract(failure):
    raw, occupancy = fixture("forces")
    raw["arguments"].update(reduction="svd", mass="lumped", backend="cudss")
    graph = raw["graphs"].pop("modes")
    raw["graphs"].update(
        {name: copy.deepcopy(graph) for name in ("weighting", "svd", "physical_lift")}
    )
    raw["graphs"]["response"]["memcpy_kinds"]["HtoD"] = 1
    if failure == "missing-lift":
        raw["graphs"].pop("physical_lift")
    elif failure == "host":
        raw["graphs"]["weighting"]["node_types_including_conditional_body"]["HOST"] = 1
    elif failure == "copy":
        raw["graphs"]["svd"]["memcpy_kinds"]["HtoD"] = 1
    elif failure == "mass":
        raw["arguments"]["mass"] = "consistent"
    elif failure == "rank":
        raw["rows"][0]["quality"]["positive"] = 3
    if failure is None:
        assert summarize(raw, occupancy, "forces")["total_seconds"] == 0.3
    else:
        with pytest.raises(ValueError):
            summarize(raw, occupancy, "forces")


def test_changed_overlay_rejected_even_with_same_wheel_version():
    raw, occ = fixture("forces")
    baseline = copy.deepcopy(summarize(raw, occ, "forces")) | {"environment": {"gpu": "same"}}
    raw["dependency"]["files"]["linear"]["sha256"] = "different implementation"
    changed = summarize(raw, occ, "forces") | {"environment": baseline["environment"]}
    with pytest.raises(ValueError, match="incompatible loaded dependencies"):
        compare(changed, baseline)
