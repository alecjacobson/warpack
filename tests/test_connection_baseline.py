"""Saved comparison gates reject incompatible or unaccepted timing evidence."""

from copy import deepcopy

import pytest

from benchmarks.applications.connection_compare import assess_baseline


def report():
    return dict(
        gpu="L40",
        warp="pinned",
        versions={"cupy": "14"},
        blas_threads="4",
        runtime_hashes={"warp_linear": "abc", "spectra_binary": "def"},
        numerical_contract_accepted=True,
        cases=[
            dict(
                method="warp-cudss",
                matrix_sha256="xyz",
                n=100,
                k=4,
                ncv=24,
                real_ncv=48,
                outer_cycles=3,
                cg_tol=1e-13,
                seed=2026,
                numerical_contract_accepted=True,
                timing_uncontended=True,
                solve_seconds=1.0,
            )
        ],
    )


@pytest.mark.parametrize("ratio,regression", [(0.7, False), (1.2, False), (1.3, True)])
def test_performance_ratio(ratio, regression):
    previous = report()
    current = deepcopy(previous)
    current["cases"][0]["solve_seconds"] = ratio
    assess_baseline(current, previous)
    assert current["baseline_comparable"]
    assert current["performance_regression"] == regression


@pytest.mark.parametrize(
    "field,value",
    [
        ("matrix_sha256", "different"),
        ("outer_cycles", 2),
        ("ncv", 32),
        ("cg_tol", 1e-11),
        ("correction_steps", 2),
        ("correction_tol", 0.1),
        ("numerical_contract_accepted", False),
        ("timing_uncontended", False),
    ],
)
def test_incompatible_row_rejected(field, value):
    previous = report()
    current = deepcopy(previous)
    current["cases"][0][field] = value
    assess_baseline(current, previous)
    assert not current["baseline_comparable"]


def test_failed_aggregate_and_changed_runtime_rejected():
    previous = report()
    for field, value in (("numerical_contract_accepted", False), ("runtime_hashes", {})):
        current = deepcopy(previous)
        current[field] = value
        assess_baseline(current, previous)
        assert not current["baseline_comparable"]
