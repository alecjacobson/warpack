"""Summarize accepted, uncontended independent Newton FFRF comparison workers."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(folder):
    rows = []
    sources = set()
    for boundary in ("free", "fixed"):
        matrices = set()
        for backend in ("newton", "cudss", "warp"):
            reports = []
            for trial in range(1, 4):
                stem = f"{boundary}-{backend}-{trial}"
                report = json.loads((folder / (stem + ".json")).read_text())
                occupancy = json.loads((folder / (stem + ".occupancy.json")).read_text())
                if not report["accepted"] or not occupancy["timing_eligible"]:
                    raise ValueError(f"{stem} is not numerically accepted and timing eligible")
                sources.add(report["source_sha256"])
                matrices.add(report["matrix_sha256"])
                reports.append(report)

            def contract(report):
                return dict(
                    matrix_sha256=report["matrix_sha256"],
                    versions=report["versions"],
                    gpu=report.get("gpu"),
                    cpu_affinity=report["cpu_affinity"],
                    threads=report["threads"],
                    settings={
                        key: report["settings"][key]
                        for key in (
                            "shape",
                            "boundary",
                            "modes",
                            "steps",
                            "inner",
                            "cycles",
                        )
                    },
                )

            comparable = contract(reports[0])
            if any(contract(r) != comparable for r in reports):
                raise ValueError("worker runtime, input or solver settings differ")
            samples = [r for report in reports for r in report["rows"]]
            if backend == "newton":
                setups = [r["setup_seconds"] for r in samples]
                checks = [r["completeness_check_seconds"] for r in samples]
                first_totals = [r["total_seconds"] for r in samples]
            else:
                setups = [r["setup_seconds"] for r in reports]
                checks = None
                first_totals = [r["setup_seconds"] + r["warmup"]["solve_seconds"] for r in reports]
            seconds = [r["solve_seconds"] for r in samples]
            batch_medians = [
                statistics.median(r["solve_seconds"] for r in x["rows"]) for x in reports
            ]
            rows.append(
                dict(
                    boundary=boundary,
                    backend=backend,
                    n=reports[0]["n"],
                    nnz=reports[0]["nnz"],
                    setup_seconds=statistics.median(setups),
                    setup_range=[min(setups), max(setups)],
                    solve_seconds=statistics.median(seconds),
                    solve_range=[min(seconds), max(seconds)],
                    batch_solve_medians=batch_medians,
                    batch_solve_max_over_min=max(batch_medians) / min(batch_medians),
                    completeness_check_seconds=statistics.median(checks) if checks else None,
                    first_use_total_seconds=statistics.median(first_totals),
                    first_use_scope="setup + solve + inertia check"
                    if checks
                    else "setup + first graph launch",
                    accuracy={
                        name: max(r["quality"][name] for r in samples)
                        for name in (
                            "relative_residual",
                            "relative_eigenvalue_error",
                            "mass_orthogonality",
                            "requested_subspace_error",
                            "rigid_overlap",
                        )
                    },
                    native_iterations=sorted({r["iterations"] for r in samples})
                    if backend != "newton"
                    else None,
                    setups=3,
                    measured_solves=len(samples),
                    accepted=True,
                    comparison_contract=comparable,
                )
            )
        if len(matrices) != 1:
            raise ValueError(f"{boundary}: matrices differ across methods")
    if len(sources) != 1:
        raise ValueError("benchmark source changed during the suite")
    environment = folder / "environment.json"
    provenance = json.loads(environment.read_text()) if environment.exists() else None
    return dict(accepted=True, source_sha256=sources.pop(), rows=rows, environment=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--regression-limit", type=float, default=1.25)
    args = parser.parse_args()
    report = summarize(args.folder)
    if args.baseline:
        prior = json.loads(args.baseline.read_text())
        if report["source_sha256"] != prior["source_sha256"]:
            raise ValueError("baseline benchmark definition differs")
        for key in ("hardware", "warp_overlay_revision", "warp_overlay_sha256"):
            if not report["environment"] or not prior.get("environment"):
                raise ValueError("baseline comparisons require recorded environment.json files")
            if report["environment"][key] != prior["environment"][key]:
                raise ValueError("baseline hardware or Warp overlay differs")
        old = {(r["boundary"], r["backend"]): r for r in prior["rows"]}
        checks = []
        for row in report["rows"]:
            baseline = old[(row["boundary"], row["backend"])]
            if row["comparison_contract"] != baseline["comparison_contract"]:
                raise ValueError("baseline matrix, runtime, affinity or solver settings differ")
            ratio = row["solve_seconds"] / baseline["solve_seconds"]
            checks.append(
                dict(
                    boundary=row["boundary"],
                    backend=row["backend"],
                    ratio=ratio,
                    accepted=ratio <= args.regression_limit,
                )
            )
        report["regression_checks"] = checks
        report["accepted"] &= all(r["accepted"] for r in checks)
    (args.folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report["rows"]:
        print(
            row["boundary"],
            row["backend"],
            "setup",
            row["setup_seconds"],
            "solve",
            row["solve_seconds"],
            "certificate",
            row["completeness_check_seconds"],
            "first use",
            row["first_use_total_seconds"],
        )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
