"""Compare Spectra contracts independently; never infer accuracy from a status label."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def contracts(row):
    """Classify only assertions actually evaluated, preserving unknown results."""
    fields = (
        ["values", "left_vectors", "right_vectors"]
        if row["family"] == "SVD"
        else ["max_absolute_residual"]
    )
    spectrum_checked = row["family"] == "SVD" or "spectrum_error" in row
    if row["family"] in ("Example1", "Example1Shift", "Example2", "Example4"):
        fields.append("spectrum_error")
    elif "spectrum_error" in row:
        fields.append("spectrum_error")
    fields.extend(key for key in ("orthogonality_error", "subspace_error") if key in row)
    tolerance = row.get("upstream_absolute_tolerance")
    evaluated = (
        row.get("outcome") in ("passed", "failed")
        and tolerance is not None
        and all(key in row for key in fields)
    )
    numerical = None
    if evaluated:
        numerical = bool(
            math.isfinite(tolerance)
            and tolerance >= 0
            and all(math.isfinite(row[key]) and 0 <= row[key] <= tolerance for key in fields)
        )
    count = row.get("reported_converged")
    count_pass = None if count is None or not evaluated else count >= row["k"]
    return dict(
        checked_numerical_bounds_pass=numerical,
        checked_fields=fields if evaluated else [],
        requested_spectrum_checked=spectrum_checked and evaluated,
        reported_count_pass=count_pass,
        upstream_allows_nonconvergence=row.get("upstream_allows_nonconvergence", False),
        # This field is deliberately stricter than numerical conformance alone.
        strict_contract_pass=numerical is True and count_pass is True,
        numerical_pass_count_short=numerical is True and count_pass is False,
        reported_convergence_space=row.get("reported_convergence_space", "unrecorded"),
    )


def index(paths):
    """Later reports override earlier cases, allowing focused regression baselines."""
    rows = {}
    for path in paths:
        cases = json.loads(Path(path).read_text())["cases"]
        if len({row["id"] for row in cases}) != len(cases):
            raise ValueError(f"Duplicate case IDs in {path}")
        rows.update({row["id"]: row for row in cases})
    return rows


def compare(baseline, current):
    changes, numerical_regressions, lost_passes = [], [], []
    contradictions = []
    classifications = {key: contracts(row) for key, row in current.items()}
    for key, row in current.items():
        contract = classifications[key]
        if row["outcome"] == "passed" and not contract["strict_contract_pass"]:
            contradictions.append(key)
        if key not in baseline:
            continue
        old = baseline[key]
        previous = contracts(old)
        if old["outcome"] != row["outcome"]:
            changes.append(dict(id=key, before=old["outcome"], after=row["outcome"]))
        if old["outcome"] == "passed" and row["outcome"] != "passed":
            lost_passes.append(key)
        if (
            previous["checked_numerical_bounds_pass"] is True
            and contract["checked_numerical_bounds_pass"] is not True
        ):
            numerical_regressions.append(key)
    missing = sorted(baseline.keys() - current.keys())
    return dict(
        current_counts=dict(Counter(row["outcome"] for row in current.values())),
        checked_numerical_counts=dict(
            Counter(
                "not_evaluated"
                if c["checked_numerical_bounds_pass"] is None
                else "passed"
                if c["checked_numerical_bounds_pass"]
                else "failed"
                for c in classifications.values()
            )
        ),
        numerical_pass_count_short=[
            key for key, c in classifications.items() if c["numerical_pass_count_short"]
        ],
        outcome_changes=changes,
        lost_passes=lost_passes,
        numerical_regressions=numerical_regressions,
        missing_cases=missing,
        new_cases=sorted(current.keys() - baseline.keys()),
        contradictory_successes=contradictions,
        regression_free=not (lost_passes or numerical_regressions or missing or contradictions),
        contracts=classifications,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare(index(args.baseline), index([args.current]))
    report.update(baseline_reports=args.baseline, current_report=args.current)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "contracts"}, indent=2))
    raise SystemExit(0 if report["regression_free"] else 1)


if __name__ == "__main__":
    main()
