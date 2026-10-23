"""Implicit covariance claims require native, original, graph and occupancy gates."""

import copy

import pytest

from benchmarks.applications.force_covariance_report import FIELDS, compare, summarize


def fixture(backend="cg"):
    q = dict.fromkeys(FIELDS, 1e-12) | dict(accepted=True, status=0, converged=4, positive=4)
    q["inner"] = (
        [dict(native=[10, 5, 0, 2], fresh=[10, 5, 1, 2], maxima=[1e-12, 1e-11]) for _ in range(2)]
        if backend == "cg"
        else [None, None]
        if backend == "cudss"
        else []
    )
    row = dict(seconds=0.1, quality=q, accepted=True, recoveries=3, exploration_guard_met=True)
    raw = dict(
        arguments=dict(backend=backend, modes=4, repeats=3, out="x", ncv=8, fixed_cycles=False),
        n=40,
        outer_capture="arpack"
        if backend == "arpack"
        else "fixed"
        if backend == "cudss"
        else "adaptive",
        accepted=True,
        prior_rank_bound=10,
        guard=0.01,
        warmup=copy.deepcopy(row),
        rows=[copy.deepcopy(row) for _ in range(3)],
        assembly_seconds=0.1,
        setup_seconds=0.2,
        capture_seconds=0 if backend == "arpack" else 0.3,
        graph=None
        if backend == "arpack"
        else dict(node_types_including_conditional_body={"KERNEL": 1}, memcpy_kinds={}),
        dependency=dict(
            warp_version="test", files={"linear": {"sha256": "same"}}, manifest_status="absent"
        ),
        source_sha256={"x": "y"},
        matrix_sha256="matrix",
        prior_sha256="prior",
        reference="oracle",
    )
    return raw, dict(timing_eligible=True, exit_code=0)


@pytest.mark.parametrize("backend", ["cg", "cudss", "arpack"])
def test_accounting_and_replay_regression(backend):
    raw, occ = fixture(backend)
    result = summarize(raw, occ) | dict(environment={"gpu": "same"})
    assert result["median_seconds"] == 0.1
    assert result["first_use_seconds"] == pytest.approx(result["setup_seconds"] + 0.1)
    assert compare(result, result) == 1
    slower = copy.deepcopy(result)
    slower["median_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(slower, result)


@pytest.mark.parametrize(
    "failure",
    [
        "aggregate",
        "contention",
        "exit",
        "repeat",
        "rank",
        "guard",
        "native",
        "positive",
        "inner",
        "fresh",
        "host",
        "dtoh",
        "htod",
        "runtime",
        "setup",
        "physical",
    ],
)
def test_invalid_answers_cannot_become_timings(failure):
    raw, occ = fixture()
    if failure == "aggregate":
        raw["accepted"] = False
    elif failure == "contention":
        occ["timing_eligible"] = False
    elif failure == "exit":
        occ["exit_code"] = 1
    elif failure == "repeat":
        raw["rows"].pop()
    elif failure == "rank":
        raw["prior_rank_bound"] = 3
    elif failure == "guard":
        raw["guard"] = float("nan")
    elif failure == "native":
        raw["rows"][0]["quality"]["converged"] = 3
    elif failure == "positive":
        raw["rows"][0]["quality"]["positive"] = 3
    elif failure == "inner":
        raw["rows"][0]["quality"]["inner"][0]["native"][2] = 1
    elif failure == "fresh":
        raw["rows"][0]["quality"]["inner"][0]["maxima"][1] = 1e-7
    elif failure == "host":
        raw["graph"]["node_types_including_conditional_body"]["HOST"] = 1
    elif failure == "dtoh":
        raw["graph"]["memcpy_kinds"]["DtoH"] = 1
    elif failure == "htod":
        raw["graph"]["memcpy_kinds"]["HtoD"] = 1
    elif failure == "runtime":
        raw["rows"][0]["seconds"] = float("nan")
    elif failure == "setup":
        raw["capture_seconds"] = -1
    elif failure == "physical":
        raw["warmup"]["quality"]["subspace_error"] = 0.1
    with pytest.raises(ValueError):
        summarize(raw, occ)


def test_only_cudss_vendor_upload_is_allowed():
    raw, occ = fixture("cudss")
    raw["graph"]["memcpy_kinds"]["HtoD"] = 2
    summarize(raw, occ)
    raw["graph"]["memcpy_kinds"]["DtoH"] = 1
    with pytest.raises(ValueError, match="transfer"):
        summarize(raw, occ)


@pytest.mark.parametrize("backend", ["cg", "cudss", "arpack"])
def test_reciprocal_requires_both_physical_equations_and_one_inverse(backend):
    raw, occ = fixture(backend)
    raw["formulation"] = "reciprocal"
    raw["arguments"]["polish_steps"] = 1
    for row in [raw["warmup"]] + raw["rows"]:
        row["quality"]["inner"] = row["quality"]["inner"][:1]
        row["quality"]["reciprocal_residual"] = 1e-10
        row["krylov_physical_converged"] = 2
        row["polish_iterations"] = 1
    result = summarize(raw, occ) | dict(environment={"gpu": "same"})
    assert result["worst_quality"]["reciprocal_residual"] == 1e-10
    assert compare(result, result) == 1
    other = copy.deepcopy(result)
    other["formulation"] = "covariance"
    with pytest.raises(ValueError, match="formulation"):
        compare(result, other)
    for field in ("residual", "reciprocal_residual"):
        bad = copy.deepcopy(raw)
        bad["warmup"]["quality"][field] = 1e-7
        with pytest.raises(ValueError, match="physical"):
            summarize(bad, occ)
    if backend != "arpack":
        bad = copy.deepcopy(raw)
        bad["rows"][0]["polish_iterations"] = None
        with pytest.raises(ValueError, match="stage metadata"):
            summarize(bad, occ)
        raw["warmup"]["quality"]["inner"] *= 2
        with pytest.raises(ValueError, match="inner"):
            summarize(raw, occ)


def test_adaptive_guard_is_not_inferred_from_full_residual_count():
    raw, occ = fixture()
    for row in [raw["warmup"]] + raw["rows"]:
        row["recoveries"] = 0
        row["exploration_guard_met"] = False
    with pytest.raises(ValueError, match="exploration guard incomplete"):
        summarize(raw, occ)
    # A deliberately fixed-budget profile is distinct; the independent spectrum
    # and subspace gates still apply, and it carries no generic completeness claim.
    raw["arguments"]["fixed_cycles"] = True
    raw["outer_capture"] = "fixed"
    result = summarize(raw, occ)
    assert result["outer_capture"] == "fixed" and not result["exploration_guard_met_all"]
    raw["warmup"]["quality"]["subspace_error"] = 0.1
    with pytest.raises(ValueError, match="physical"):
        summarize(raw, occ)


@pytest.mark.parametrize("failure", ["missing", "policy", "counts", "nan"])
def test_outer_policy_metadata_is_required(failure):
    raw, occ = fixture()
    if failure == "missing":
        raw.pop("outer_capture")
    elif failure == "policy":
        raw["outer_capture"] = "fixed"
    elif failure == "counts":
        raw["rows"][0]["recoveries"] = 0
    else:
        raw["rows"][0]["recoveries"] = float("nan")
    with pytest.raises(ValueError):
        summarize(raw, occ)


@pytest.mark.parametrize(
    "key",
    [
        "source_sha256",
        "matrix_sha256",
        "prior_sha256",
        "environment",
        "reference",
        "dependency",
        "arguments",
    ],
)
def test_incompatible_baselines(key):
    raw, occ = fixture()
    base = summarize(raw, occ) | dict(environment={"gpu": "same"})
    changed = copy.deepcopy(base)
    if key == "dependency":
        changed[key]["warp_version"] = "changed"
    elif isinstance(changed[key], dict):
        changed[key]["changed"] = True
    else:
        changed[key] = "changed"
    with pytest.raises(ValueError, match="incompatible"):
        compare(changed, base)
