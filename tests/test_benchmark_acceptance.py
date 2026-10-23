"""A passing primary solver cannot hide failed requested comparison evidence."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.acceptance import assess_comparison, errors_below_tolerance, require_accepted


@pytest.mark.parametrize(
    "failure",
    [
        dict(accepted=False),
        dict(outcome="unsupported"),
        dict(outcome="error"),
        dict(outcome="timeout"),
        dict(outcome="worker_error"),
        dict(strict_counts_pass=False),
        dict(status=2),
        dict(validation=[dict(accepted=True, solver_full_count=False)]),
        dict(validation=[dict(accepted=True, native_stopping_passed=False)]),
        dict(validation=[dict(accepted=True, solver_status=3)]),
        dict(validation=[]),
        dict(full_block_absolute_tolerance_met=False),
        dict(cg_statistics=[dict(unconverged_rhs=1)]),
    ],
)
def test_comparison_rejects_every_failed_requested_contract(failure):
    row = dict(accepted=True, outcome="passed", validation=[dict(accepted=True)])
    row.update(failure)
    old = deepcopy(row)
    report = {}
    rows = [dict(accepted=True, validation=[dict(accepted=True)]), row]
    assert not assess_comparison(report, rows)
    with pytest.raises(SystemExit, match="Rejected requested comparison"):
        require_accepted(report)
    assert rows[0]["numerical_contract_accepted"]
    assert not row["numerical_contract_accepted"]
    assert all(row[key] == value for key, value in old.items())
    assert len(rows) == 2


def test_diagnostic_preserves_failures_and_empty_report_is_rejected():
    report = {}
    assert not assess_comparison(report, [])
    with pytest.raises(SystemExit):
        require_accepted(report)
    row = dict(accepted=False, outcome="unsupported", reason="missing reference API")
    assert not assess_comparison(report, [row], diagnostic=True)
    require_accepted(report)
    assert report["diagnostic"] and not report["numerical_contract_accepted"]
    assert not row["accepted"] and row["outcome"] == "unsupported"


def test_timing_and_graph_are_separate_from_numerical_acceptance():
    row = dict(
        accepted=True,
        validation=[dict(accepted=True)],
        timing_uncontended=False,
        device_only_graph=False,
    )
    report = {}
    assert assess_comparison(report, [row])
    require_accepted(report)
    assert not row["timing_uncontended"] and not row["device_only_graph"]


@pytest.mark.parametrize("failure", ["error", "unsupported", "worker_error", "native", "none"])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_parent_cli_preserves_requested_worker_result(monkeypatch, tmp_path, failure, diagnostic):
    from benchmarks.applications import compare

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "report.json"
    argv = ["compare", "--classes", "A", "--sizes", "80", "--methods", "cupy", "--out", str(output)]
    if diagnostic:
        argv.append("--diagnostic")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(compare.wp, "get_device", lambda: SimpleNamespace(name="test GPU"))
    monkeypatch.setattr(compare.subprocess, "check_output", lambda *a, **k: "")

    def worker(command, **kwargs):
        if failure == "worker_error":
            raise subprocess.CalledProcessError(1, command)
        row = dict(
            api_class="A",
            requested_size=80,
            method="cupy",
            accepted=True,
            outcome="passed",
            validation=[dict(accepted=True, native_stopping_passed=True)],
        )
        if failure in ("error", "unsupported"):
            row.update(accepted=False, outcome=failure, reason="deliberate test result")
        elif failure == "native":
            row["validation"][0]["native_stopping_passed"] = False
        Path(command[command.index("--out") + 1]).write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare.subprocess, "run", worker)
    if failure != "none" and not diagnostic:
        with pytest.raises(SystemExit):
            compare.main()
    else:
        compare.main()
    saved = json.loads(output.read_text())
    assert len(saved["cases"]) == 1
    assert saved["numerical_contract_accepted"] == (failure == "none")
    assert saved["diagnostic"] == diagnostic
    if failure in ("error", "unsupported", "worker_error"):
        assert saved["cases"][0]["outcome"] == failure


def test_success_flag_without_trial_evidence_is_rejected():
    report = {}
    assert not assess_comparison(report, [dict(accepted=True, outcome="passed")])


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), -1e-30, 1e-12, 2e-12]
)
def test_error_vector_rejects_invalid_measurement_at_every_position(position, invalid):
    errors = [0.0, 1e-15, 9e-13]
    errors[position] = invalid
    assert not errors_below_tolerance(errors, 1e-12)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -float("inf"), -1.0, 0.0])
def test_error_vector_rejects_invalid_threshold(tolerance):
    assert not errors_below_tolerance([0.0], tolerance)


def test_error_vector_requires_evidence_and_keeps_strict_original_threshold():
    assert not errors_below_tolerance([], 1e-12)
    assert errors_below_tolerance([0.0, 1e-15, 9e-13], 1e-12)
    assert errors_below_tolerance(iter([0.0, 1e-15]), 1e-12)
    assert not errors_below_tolerance([0.0, 1e-12], 1e-12)


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -1e-30, 2e-12])
def test_inclusive_error_vector_rejects_invalid_measurement(position, invalid):
    errors = [0.0, 1e-15, 1e-12]
    errors[position] = invalid
    assert not errors_below_tolerance(errors, 1e-12, inclusive=True)


def test_inclusive_error_vector_preserves_equality_and_rejects_missing_evidence():
    assert errors_below_tolerance([0.0, 1e-12], 1e-12, inclusive=True)
    assert not errors_below_tolerance([], 1e-12, inclusive=True)
