"""Signed buckling timing identities must not mix loads or ignore failed gates."""

from copy import deepcopy

import pytest

from benchmarks.applications.buckling_compare import assess_baseline


def report():
    return dict(
        args=dict(side=16, shift=0.1, ncv=64, methods="warp-cg"),
        gpu="L40",
        versions={"warp": "test"},
        blas_threads="4",
        upstream_source_sha256={"linear": "same"},
        numerical_contract_accepted=True,
        timing_eligible=True,
        cases=[
            dict(
                sign=sign,
                method="warp-cg",
                accepted=True,
                n=41616,
                k=20,
                sigma=sigma,
                matrix_sha256="same-pencil",
                solve_seconds=seconds,
            )
            for sign, sigma, seconds in [("positive", 0.1, 1.0), ("negative", -0.1, 2.0)]
        ],
    )


def test_signed_rows_match_their_own_baseline():
    old = report()
    new = deepcopy(old)
    new["cases"][1]["solve_seconds"] = 3.0
    new["cases"].reverse()
    assess_baseline(new, old)
    assert new["baseline_comparable"] and new["performance_regression"]
    assert new["cases"][0]["time_ratio"] == 1.5
    assert new["cases"][1]["time_ratio"] == 1.0


@pytest.mark.parametrize(
    "change", ["matrix", "tolerance", "runtime", "missing", "rejected", "contention"]
)
def test_changed_or_unaccepted_baseline_is_not_timing_evidence(change):
    old = report()
    new = deepcopy(old)
    if change == "matrix":
        new["cases"][0]["matrix_sha256"] = "different"
    if change == "tolerance":
        new["args"]["outer_tol"] = 1e-8
    if change == "runtime":
        new["versions"]["warp"] = "different"
    if change == "missing":
        old["cases"].pop()
    if change == "rejected":
        old["numerical_contract_accepted"] = False
    if change == "contention":
        new["timing_eligible"] = False
    assess_baseline(new, old)
    assert not new["baseline_comparable"]


def test_requested_subset_can_use_a_complete_baseline():
    old = report()
    new = deepcopy(old)
    new["cases"] = new["cases"][:1]
    new["args"]["signs"] = "positive"
    assess_baseline(new, old)
    assert new["baseline_comparable"] and not new["performance_regression"]
