"""CPU-only status evidence must not be hidden by accurate eigenpairs."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.upstream.compare_reports import compare, contracts
from benchmarks.upstream.run_spectra import solver_status


def row(**changes):
    return dict(
        dict(
            id="case",
            family="SymEigs",
            outcome="passed",
            k=1,
            upstream_absolute_tolerance=1e-9,
            max_absolute_residual=1e-14,
            reported_converged=1,
        ),
        **changes,
    )


@pytest.mark.parametrize("status", [1, -1, float("nan"), float("inf")])
def test_failure_status_rejects_accurate_complete_result(status):
    result = contracts(row(solver_status=status))
    assert result["checked_numerical_bounds_pass"]
    assert result["reported_count_pass"]
    assert result["reported_status_pass"] is False
    assert not result["strict_contract_pass"]
    assert compare({}, {"case": row(solver_status=status)})["contradictory_successes"] == ["case"]


def test_historical_missing_status_stays_unknown():
    result = contracts(row())
    assert result["reported_status_pass"] is None
    assert result["strict_contract_pass"]  # Preserve historical classification, not a status claim.
    assert contracts(row(solver_status=0))["reported_status_pass"] is True
    assert not contracts(row(solver_status_required=True))["strict_contract_pass"]


@pytest.mark.parametrize("status", [1, None])
def test_status_regression_is_visible_even_when_both_rows_failed(status):
    old = row(outcome="failed", reported_converged=0, solver_status=0)
    new = row(outcome="failed", reported_converged=0, solver_status=status)
    report = compare({"case": old}, {"case": new})
    assert not report["outcome_changes"]
    assert report["status_regressions"] == ["case"]
    assert not report["regression_free"]


def test_wrapper_status_extraction_without_gpu_work():
    device_array = SimpleNamespace(numpy=lambda: np.array([3], dtype=np.int32))
    leaf = SimpleNamespace(status=device_array)
    assert solver_status(leaf) == 3
    assert solver_status(SimpleNamespace(core=leaf)) == 3
    assert solver_status(SimpleNamespace(real=SimpleNamespace(core=leaf))) == 3
    assert solver_status(SimpleNamespace()) is None
    cycle = SimpleNamespace()
    cycle.core = cycle
    assert solver_status(cycle) is None
