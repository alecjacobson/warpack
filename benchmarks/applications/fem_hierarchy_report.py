"""Reject incomplete/contended hierarchy comparisons and gate warm-path regressions."""

import argparse
import json
import math
import statistics
from pathlib import Path

STRATEGIES = ("random", "coarse", "retained")


def summarize(report, occupancy):
    if not occupancy["timing_eligible"] or occupancy["exit_code"] != 0 or not report["accepted"]:
        raise ValueError("rejected or contended run")
    args = report["arguments"]
    if len(report["rows"]) != args["repeats"] or args["repeats"] < 3:
        raise ValueError("need at least three complete repeats")
    if set(report["graphs"]) != {"fine", "coarse", "transfer_refine"}:
        raise ValueError("incomplete graph audit")
    for graph in report["graphs"].values():
        if "HOST" in graph["node_types_including_conditional_body"] or any(
            kind.startswith(("HtoD", "DtoH")) for kind in graph["memcpy_kinds"]
        ):
            raise ValueError("host operations in graph")
    for sequence in [report["warmup"]] + report["rows"]:
        if [r["strategy"] for r in sequence] != list(STRATEGIES):
            raise ValueError("incomplete or mislabeled sequence")
        for r in sequence:
            fields = ("coarse_seconds", "transfer_seconds", "solve_seconds", "total_seconds")
            if any(not math.isfinite(r[f]) or r[f] < 0 for f in fields) or r["solve_seconds"] <= 0:
                raise ValueError("invalid timings")
            if not math.isclose(r["total_seconds"], sum(r[f] for f in fields[:3]), rel_tol=1e-12):
                raise ValueError("inconsistent runtime total")
            if (r["coarse_quality"] is not None) != (r["strategy"] == "coarse"):
                raise ValueError("missing or unexpected coarse verification")
            for q in (r["quality"], r["coarse_quality"]):
                if q is None:
                    continue
                if not q["accepted"] or q["status"] != 0 or q["converged"] != args["modes"]:
                    raise ValueError("failed numerical/native checks")
                for key in (
                    "residual",
                    "orthogonality",
                    "spectrum_error",
                    "requested_subspace_error",
                ):
                    if not math.isfinite(q[key]) or not 0 <= q[key] < 1e-8:
                        raise ValueError("failed physical accuracy checks")
    setup = report["setup"]
    if any(not math.isfinite(t) or t <= 0 for t in setup.values()):
        raise ValueError("invalid setup timings")
    rows = []
    for i, strategy in enumerate(STRATEGIES):
        trials = [seq[i] for seq in report["rows"]]
        first = report["warmup"][i]
        # Component accounting, not an independently repeated cold-process total.
        construction = setup["fine_assembly_seconds"] + setup["fine_setup_capture_seconds"]
        if strategy == "coarse":
            construction += (
                setup["coarse_assembly_seconds"]
                + setup["coarse_setup_capture_seconds"]
                + setup["transfer_setup_capture_seconds"]
            )
        rows.append(
            dict(
                strategy=strategy,
                setup_assembly_seconds=construction if strategy != "retained" else None,
                first_use_component_seconds=construction + first["total_seconds"]
                if strategy != "retained"
                else None,
                total_seconds=statistics.median(r["total_seconds"] for r in trials),
                coarse_seconds=statistics.median(r["coarse_seconds"] for r in trials),
                refine_seconds=statistics.median(r["solve_seconds"] for r in trials),
                iterations=[
                    min(r["quality"]["iterations"] for r in trials),
                    max(r["quality"]["iterations"] for r in trials),
                ],
                residual=max(r["quality"]["residual"] for r in trials),
                subspace_error=max(r["quality"]["requested_subspace_error"] for r in trials),
            )
        )
    signature = {
        key: report[key]
        for key in ("source_sha256", "matrix_sha256", "warp", "versions", "dimensions")
    }
    signature["configuration"] = {
        key: args[key] for key in ("coarse", "fine", "modes", "width", "mass", "repeats")
    }
    return dict(signature=signature, rows=rows)


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("regression limit must be finite and >=1")
    for key in ("signature", "environment"):
        if current[key] != baseline[key]:
            raise ValueError("incompatible baseline " + key)
    old = {r["strategy"]: r for r in baseline["rows"]}
    if set(old) != set(STRATEGIES) or {r["strategy"] for r in current["rows"]} != set(STRATEGIES):
        raise ValueError("incompatible baseline rows")
    ratios = {
        r["strategy"]: r["total_seconds"] / old[r["strategy"]]["total_seconds"]
        for r in current["rows"]
    }
    if any(r > limit for r in ratios.values()):
        raise ValueError(f"warm workflow regression above {limit}: {ratios}")
    return ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.report.read_text())
    result = summarize(raw, json.loads(args.report.with_suffix(".occupancy.json").read_text()))
    result["environment"] = json.loads((args.report.parent / "environment.json").read_text())
    if args.baseline:
        result["baseline_ratios"] = compare(result, json.loads(args.baseline.read_text()))
    args.report.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(row)


if __name__ == "__main__":
    main()
