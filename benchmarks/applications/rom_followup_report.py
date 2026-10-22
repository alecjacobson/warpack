"""Strict acceptance and reproducible timing summaries for ROM follow-up runs."""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks.dependency_provenance import signature


def quality(q, k, fields, positive=False):
    if not q["accepted"] or q["status"] != 0 or q["converged"] != k:
        raise ValueError("failed native/aggregate acceptance")
    if positive and q["positive"] != k:
        raise ValueError("insufficient positive covariance rank")
    for f in fields:
        if q[f] is None or not math.isfinite(q[f]) or not 0 <= q[f] < 1e-8:
            raise ValueError("invalid physical accuracy: " + f)


def summarize(raw, occupancy, kind):
    if kind not in ("hierarchy", "forces"):
        raise ValueError("unknown report kind")
    if not raw["accepted"] or not occupancy["timing_eligible"] or occupancy["exit_code"]:
        raise ValueError("failed or contended worker")
    args = raw["arguments"]
    if args["repeats"] < 3 or len(raw["rows"]) != args["repeats"]:
        raise ValueError("need at least three complete repeats")
    if kind == "hierarchy":
        timings = ("source_seconds", "fine_seconds")
        fields = ("residual", "orthogonality", "spectrum_error", "requested_subspace_error")
        setup = raw["setup"]
    else:
        timings = ("response_seconds", "modes_seconds")
        fields = (
            "source_residual",
            "residual",
            "orthogonality",
            "spectrum_error",
            "subspace_error",
            "energy_error",
        )
        setup = {k: raw[k] for k in ("assembly_seconds", "setup_seconds", "capture_seconds")}
    if any(not math.isfinite(x) or x <= 0 for x in setup.values()):
        raise ValueError("invalid setup timing")
    for row in [raw["warmup"]] + raw["rows"]:
        quality(row["quality"], args["modes"], fields, positive=kind == "forces")
        if kind == "forces" and args["backend"] == "cg":
            if row["quality"]["cg_stats"] is None or row["quality"]["cg_stats"][2] != 0:
                raise ValueError("incomplete source CG solve")
        if not all(math.isfinite(row[t]) and row[t] > 0 for t in (*timings, "total_seconds")):
            raise ValueError("invalid runtime")
        if not math.isclose(row["total_seconds"], sum(row[t] for t in timings), rel_tol=1e-12):
            raise ValueError("invalid total")
        if kind == "hierarchy":
            if not row["accepted"]:
                raise ValueError("failed hierarchy")
            source = row["source_quality"]
            if args["profile"] == "random":
                if source is not None or raw["source_count"] is not None:
                    raise ValueError("unexpected coarse source")
            else:
                full = args["profile"].startswith("full-")
                tol = 1e-3 if args["profile"].endswith("loose") else 1e-11
                count = args["width"] if full else args["modes"]
                if (
                    raw["source_tolerance"] != tol
                    or raw["source_count"] != count
                    or source is None
                    or source["status"] != 0
                    or source["converged"] != count
                ):
                    raise ValueError("invalid coarse source")
                if (
                    not math.isfinite(source["residual"])
                    or not 0 <= source["residual"] <= 1.01 * tol
                    or not math.isfinite(source["orthogonality"])
                    or not 0 <= source["orthogonality"] < 1e-8
                ):
                    raise ValueError("invalid coarse physical checks")
    expected_graphs = {"response", "modes"} if kind == "forces" else {"fine"}
    if kind == "forces" and args.get("reduction", "gram") == "svd":
        if args["mass"] != "lumped":
            raise ValueError("weighted-SVD composition requires lumped mass")
        expected_graphs = {"response", "weighting", "svd", "physical_lift"}
    if kind == "hierarchy" and args["profile"] != "random":
        expected_graphs.add("coarse")
    if set(raw["graphs"]) != expected_graphs:
        raise ValueError("missing or unexpected graph audit")
    for name, graph in raw["graphs"].items():
        if not graph["node_types_including_conditional_body"].get("KERNEL", 0):
            raise ValueError("empty graph audit")
        if "HOST" in graph["node_types_including_conditional_body"]:
            raise ValueError("host callback in graph")
        for copy in graph["memcpy_kinds"]:
            vendor = kind == "forces" and args["backend"] == "cudss" and name == "response"
            if copy.startswith("DtoH") or (copy.startswith("HtoD") and not vendor):
                raise ValueError("unexpected host transfer")
    return dict(
        dependency=raw.get("dependency"),
        arguments=args,
        setup_seconds=sum(setup.values()),
        first_use_seconds=sum(setup.values()) + raw["warmup"]["total_seconds"],
        stages={t: statistics.median(r[t] for r in raw["rows"]) for t in timings},
        total_seconds=statistics.median(r["total_seconds"] for r in raw["rows"]),
        worst_quality={
            f: max(r["quality"][f] for r in [raw["warmup"]] + raw["rows"]) for f in fields
        },
        source_sha256=raw["source_sha256"],
        matrix_sha256=raw["matrix_sha256"],
        forces_sha256=raw["forces_sha256"] if kind == "forces" else None,
    )


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("invalid regression limit")
    if signature(current.get("dependency")) != signature(baseline.get("dependency")):
        raise ValueError("incompatible loaded dependencies")
    # Exact source identity is deliberate: revised implementations need an
    # explicitly reviewed baseline, not silent mixing of benchmark definitions.
    for key in ("source_sha256", "matrix_sha256", "forces_sha256", "environment"):
        if current[key] != baseline[key]:
            raise ValueError("incompatible baseline " + key)
    current_args = {k: v for k, v in current["arguments"].items() if k != "out"}
    old_args = {k: v for k, v in baseline["arguments"].items() if k != "out"}
    if current_args != old_args:
        raise ValueError("incompatible configurations")
    ratio = current["total_seconds"] / baseline["total_seconds"]
    if not math.isfinite(ratio) or ratio > limit:
        raise ValueError("workflow regression")
    return ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--kind", choices=("hierarchy", "forces"), required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = summarize(
        json.loads(args.report.read_text()),
        json.loads(args.report.with_suffix(".occupancy.json").read_text()),
        args.kind,
    )
    report["environment"] = json.loads((args.report.parent / "environment.json").read_text())
    if args.baseline:
        report["baseline_ratio"] = compare(report, json.loads(args.baseline.read_text()))
    args.report.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
