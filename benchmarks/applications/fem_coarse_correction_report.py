"""Accuracy, provenance and timing gates for the experimental coarse correction."""

import argparse
import json
import math
import statistics
from pathlib import Path

from .fem_hierarchy_tuning import source_usable


def summarize(raw, occupancy, exclusion=None):
    if exclusion is not None:
        raise ValueError(f"manually excluded timing: {exclusion['reason']}")
    if not raw["accepted"] or not occupancy["timing_eligible"] or occupancy["exit_code"]:
        raise ValueError("failed or contended run")
    args = raw["arguments"]
    if len(raw["rows"]) != args["repeats"] or args["repeats"] < 3:
        raise ValueError("at least three complete measured repeats required")
    coarse = args["profile"] not in ("cg", "cg64", "fsai") and not args["profile"].startswith(
        "geometric"
    )
    if set(raw["graphs"]) != ({"fine", "coarse"} if coarse else {"fine"}):
        raise ValueError("missing graph audit")
    for graph in raw["graphs"].values():
        if "HOST" in graph["node_types_including_conditional_body"] or any(
            k.startswith(("HtoD", "DtoH")) for k in graph["memcpy_kinds"]
        ):
            raise ValueError("host work in captured graph")
    for row in [raw["warmup"], *raw["rows"]]:
        if not row["accepted"] or row["correction_status"]:
            raise ValueError("failed coarse correction")
        q = row["quality"]
        if not q["accepted"] or q["status"] or q["converged"] != args["modes"]:
            raise ValueError("native or numerical failure")
        for key in ("residual", "orthogonality", "spectrum_error", "requested_subspace_error"):
            if not math.isfinite(q[key]) or not 0 <= q[key] < 1e-8:
                raise ValueError("failed fine accuracy")
        cq = row["source_quality"]
        if coarse:
            if (
                cq is None
                or raw["source_count"] != args["width"]
                or raw["source_tolerance"] != args["coarse_tol"]
                or not source_usable(cq, args["width"], args["coarse_tol"])
            ):
                raise ValueError("invalid coarse source")
        elif cq is not None or row["source_seconds"] != 0:
            raise ValueError("unexpected coarse source")
        for key in ("source_seconds", "fine_seconds", "total_seconds"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("invalid timing")
        if row["fine_seconds"] <= 0 or not math.isclose(
            row["total_seconds"], row["source_seconds"] + row["fine_seconds"], rel_tol=1e-12
        ):
            raise ValueError("invalid workflow total")
    setup = raw["setup"]
    expected = {"fine_assembly_seconds", "fine_setup_seconds", "fine_capture_seconds"}
    if coarse:
        expected |= {
            "coarse_assembly_seconds",
            "coarse_setup_capture_seconds",
            "coarse_initial_solve_seconds",
            "transfer_setup_seconds",
        }
    if set(setup) != expected or any(not math.isfinite(v) or v <= 0 for v in setup.values()):
        raise ValueError("incomplete setup accounting")
    if not math.isclose(raw["setup_total_seconds"], sum(setup.values()), rel_tol=1e-12):
        raise ValueError("invalid setup total")
    result = {
        key: statistics.median(r[key] for r in raw["rows"])
        for key in ("source_seconds", "fine_seconds", "total_seconds")
    }
    result.update(
        setup_total_seconds=raw["setup_total_seconds"],
        first_use_component_seconds=raw["setup_total_seconds"] + raw["warmup"]["total_seconds"],
        memory_bytes=raw["setup_capture_device_free_memory_decrease_bytes"],
        iterations=[r["quality"]["iterations"] for r in raw["rows"]],
        residual=max(r["quality"]["residual"] for r in raw["rows"]),
        subspace_error=max(r["quality"]["requested_subspace_error"] for r in raw["rows"]),
        signature={
            k: raw[k] for k in ("matrix_sha256", "source_sha256", "dependency", "environment", "n")
        },
    )
    result["signature"]["arguments"] = {k: v for k, v in args.items() if k != "out"}
    return result


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("regression limit must be finite and >=1")
    if current["signature"] != baseline["signature"]:
        raise ValueError("incompatible baseline")
    ratio = current["total_seconds"] / baseline["total_seconds"]
    if not math.isfinite(ratio) or ratio > limit:
        raise ValueError(f"warm workflow regression: {ratio}")
    return ratio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    summary = summarize(
        json.loads(args.report.read_text()),
        json.loads(args.report.with_suffix(".occupancy.json").read_text()),
        json.loads(args.report.with_suffix(".exclusion.json").read_text())
        if args.report.with_suffix(".exclusion.json").exists()
        else None,
    )
    if args.baseline:
        summary["baseline_ratio"] = compare(summary, json.loads(args.baseline.read_text()))
    args.report.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "signature"}, indent=2))


if __name__ == "__main__":
    main()
