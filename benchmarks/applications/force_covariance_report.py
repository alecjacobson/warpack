"""Strict physical/native, graph and timing gates for implicit covariance runs."""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks.dependency_provenance import signature

FIELDS = ("residual", "orthogonality", "spectrum_error", "subspace_error", "captured_energy_error")


def summarize(raw, occupancy):
    if not raw["accepted"] or not occupancy["timing_eligible"] or occupancy["exit_code"]:
        raise ValueError("failed or contended worker")
    args = raw["arguments"]
    backend = args["backend"]
    k = args["modes"]
    policy = raw.get("outer_capture")
    expected_policy = (
        "arpack"
        if backend == "arpack"
        else ("fixed" if backend == "cudss" or args.get("fixed_cycles") else "adaptive")
    )
    if policy != expected_policy:
        raise ValueError("missing or inconsistent outer capture policy")
    if (
        backend not in ("arpack", "cg", "cudss")
        or args["repeats"] < 3
        or len(raw["rows"]) != args["repeats"]
    ):
        raise ValueError("invalid method or repeat count")
    if raw["prior_rank_bound"] < k or not math.isfinite(raw["guard"]) or raw["guard"] <= 0:
        raise ValueError("invalid reference rank/guard")
    for row in [raw["warmup"]] + raw["rows"]:
        if not row["accepted"]:
            raise ValueError("rejected outer solve")
        if policy != "arpack":
            if not isinstance(row.get("recoveries"), int) or row["recoveries"] < 0:
                raise ValueError("invalid exploration metadata")
            guard = args["ncv"] >= raw["n"] or row["recoveries"] + 1 >= k
            if row["exploration_guard_met"] != guard:
                raise ValueError("invalid exploration metadata")
            if policy == "adaptive" and not guard:
                raise ValueError("adaptive exploration guard incomplete")
        q = row["quality"]
        if not q["accepted"] or q["status"] != 0 or q["converged"] != k or q["positive"] != k:
            raise ValueError("incomplete native/positive modes")
        if any(not math.isfinite(q[f]) or not 0 <= q[f] < 1e-8 for f in FIELDS):
            raise ValueError("invalid physical accuracy")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("invalid replay timing")
        if backend == "cg":
            if len(q["inner"]) != 2 or any(
                s is None
                or s["native"][2] != 0
                or not math.isfinite(s["maxima"][1])
                or not 0 <= s["maxima"][1] < 1e-8
                for s in q["inner"]
            ):
                raise ValueError("incomplete or inaccurate inner solve")
        elif q["inner"] != ([None, None] if backend == "cudss" else []):
            raise ValueError("unexpected inner metadata")
    setup = [raw[k] for k in ("assembly_seconds", "setup_seconds", "capture_seconds")]
    if any(not math.isfinite(x) or x < 0 for x in setup) or min(setup[:2]) <= 0:
        raise ValueError("invalid setup timing")
    graph = raw["graph"]
    if backend == "arpack":
        if graph is not None or setup[2] != 0:
            raise ValueError("unexpected CPU graph")
    else:
        if (
            setup[2] <= 0
            or graph is None
            or not graph["node_types_including_conditional_body"].get("KERNEL", 0)
            or graph["node_types_including_conditional_body"].get("HOST", 0)
        ):
            raise ValueError("invalid GPU graph")
        if any(
            "DtoH" in kind or (backend == "cg" and "HtoD" in kind) for kind in graph["memcpy_kinds"]
        ):
            raise ValueError("unexpected host transfer")
    return dict(
        arguments=args,
        dependency=raw["dependency"],
        source_sha256=raw["source_sha256"],
        matrix_sha256=raw["matrix_sha256"],
        prior_sha256=raw["prior_sha256"],
        reference=raw["reference"],
        outer_capture=policy,
        exploration_guard_met_all=None
        if policy == "arpack"
        else all(row["exploration_guard_met"] for row in [raw["warmup"]] + raw["rows"]),
        setup_seconds=sum(setup),
        first_use_seconds=sum(setup) + raw["warmup"]["seconds"],
        median_seconds=statistics.median(r["seconds"] for r in raw["rows"]),
        seconds=[r["seconds"] for r in raw["rows"]],
        worst_quality={
            f: max(r["quality"][f] for r in [raw["warmup"]] + raw["rows"]) for f in FIELDS
        },
    )


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("invalid regression limit")
    if signature(current["dependency"]) != signature(baseline["dependency"]):
        raise ValueError("incompatible dependency")
    for key in ("source_sha256", "matrix_sha256", "prior_sha256", "environment", "reference"):
        if current[key] != baseline[key]:
            raise ValueError("incompatible " + key)
    if {k: v for k, v in current["arguments"].items() if k != "out"} != {
        k: v for k, v in baseline["arguments"].items() if k != "out"
    }:
        raise ValueError("incompatible configuration")
    ratio = current["median_seconds"] / baseline["median_seconds"]
    if not math.isfinite(ratio) or ratio > limit:
        raise ValueError("replay regression")
    return ratio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    result = summarize(
        json.loads(args.report.read_text()),
        json.loads(args.report.with_suffix(".occupancy.json").read_text()),
    )
    result["environment"] = json.loads((args.report.parent / "environment.json").read_text())
    if args.baseline:
        result["baseline_ratio"] = compare(result, json.loads(args.baseline.read_text()))
    args.report.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
