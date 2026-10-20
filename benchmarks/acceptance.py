"""Process contracts for composite benchmarks; numerical evidence stays intact."""


def add_diagnostic_argument(parser):
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="retain rejected/unsupported requested methods without failing the numerical gate; timing and baseline gates still apply",
    )


def assess_comparison(report, rows, diagnostic=False):
    """Check recorded numerical/stopping contracts without rewriting old gates.

    Graph compliance and timing eligibility remain separate. Only declared
    stopping fields are checked; no new residual threshold is inferred.
    Internal workers transport rows even when rejected; the parent enforces
    the requested-method contract after retaining the complete report.
    """
    for row in rows:
        failures = []
        if not row.get("accepted", False):
            failures.append("physical acceptance")
        if row.get("outcome", "passed") != "passed":
            failures.append("requested method " + row["outcome"])
        for key in (
            "strict_counts_pass",
            "strict_stopping_accepted",
            "inner_solve_accepted",
            "full_block_absolute_tolerance_met",
        ):
            if key in row and not row[key]:
                failures.append(key)
        for status in ("status", "solver_status"):
            if row.get(status) is not None and row[status] != 0:
                failures.append(status)
        checks = row.get("validation", row.get("checks"))
        if not checks:
            failures.append("no trial evidence")
        for check in checks or []:
            if "accepted" in check and not check["accepted"]:
                failures.append("trial physical acceptance")
            for key in ("solver_full_count", "native_stopping_passed"):
                if key in check and not check[key]:
                    failures.append(key)
            for status in ("status", "solver_status"):
                if check.get(status) is not None and check[status] != 0:
                    failures.append("trial " + status)
        # These statistics belong to actual inverse solves in application run(),
        # not the intentionally truncated block preconditioners handled in #84.
        if any(s.get("unconverged_rhs", 0) for s in row.get("cg_statistics", [])):
            failures.append("inner inverse convergence")
        row["numerical_contract_failures"] = sorted(set(failures))
        row["numerical_contract_accepted"] = not failures
    report["numerical_contract_accepted"] = bool(rows) and all(
        row["numerical_contract_accepted"] for row in rows
    )
    report["diagnostic"] = bool(diagnostic)
    report["process_acceptance_policy"] = (
        "Every requested row must pass its recorded physical and stopping gates; "
        "unsupported/error/timeout rows are rejected. Diagnostic mode preserves "
        "these failures without failing the numerical process gate. Graph, timing "
        "and baseline-regression results remain separate."
    )
    return report["numerical_contract_accepted"]


def require_accepted(report):
    if not report["numerical_contract_accepted"] and not report["diagnostic"]:
        raise SystemExit("Rejected requested comparison; all numerical evidence retained")
