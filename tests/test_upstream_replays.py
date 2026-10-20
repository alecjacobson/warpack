"""A later passing graph replay cannot erase earlier numerical/status failures."""

import pytest

from benchmarks.upstream.compare_reports import contracts
from benchmarks.upstream.run_spectra import combine_accuracy_trials

CASE = dict(family="SymEigs", k=2)


def passed():
    return dict(
        outcome="passed",
        upstream_absolute_tolerance=1e-9,
        max_absolute_residual=1e-12,
        spectrum_error=1e-12,
        reported_converged=2,
        returned_pairs=2,
        solver_status=0,
        solver_status_required=True,
        original_evaluation_converged=2,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_absolute_residual", 1e-4),
        ("spectrum_error", 1e-3),
        ("reported_converged", 1),
        ("solver_status", 1),
        ("solver_status", None),
        ("max_absolute_residual", float("nan")),
    ],
)
def test_first_failed_last_passed_replay_is_rejected(field, value):
    first, last = passed(), passed()
    first[field] = value
    # Leave the synthetic first status label at passed: numerical assertions
    # must remain authoritative even if a producer mislabeled the trial.
    result = combine_accuracy_trials(CASE, [first, last])
    assert result["outcome"] == "failed" and not result["all_replays_pass"]
    assert not contracts(dict(**CASE, **result))["strict_contract_pass"]
    assert result["validated_replays"] == 2
    assert result["accuracy_trials"] == [first, last]
    assert last == passed()


def test_all_passing_replays_retain_worst_assertions():
    first, last = passed(), passed()
    first["max_absolute_residual"] = 3e-12
    last["spectrum_error"] = 2e-12
    result = combine_accuracy_trials(CASE, [first, last])
    assert result["outcome"] == "passed" and result["all_replays_pass"]
    assert result["max_absolute_residual"] == 3e-12
    assert result["spectrum_error"] == 2e-12
    assert result["reported_status_pass"] and result["reported_count_pass"]
    with pytest.raises(ValueError, match="at least one"):
        combine_accuracy_trials(CASE, [])


def test_paired_svd_count_is_required_on_every_replay():
    first = dict(
        outcome="failed",
        upstream_absolute_tolerance=1e-9,
        values=0.0,
        left_vectors=0.0,
        right_vectors=0.0,
        reported_converged=2,
        left_converged=1,
        right_converged=2,
        solver_status=0,
        solver_status_required=True,
    )
    last = dict(first, left_converged=2, outcome="passed")
    case = dict(family="SVD", k=2)
    result = combine_accuracy_trials(case, [first, last])
    assert result["outcome"] == "failed" and result["left_converged"] == 1
    assert not contracts(dict(**case, **result))["physical_count_pass"]
