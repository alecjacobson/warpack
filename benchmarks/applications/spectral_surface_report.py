"""Reject incomplete, inaccurate or contended dominant-pencil timing claims."""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks.dependency_provenance import signature as dependency_signature


def summarize(raw, occupancy):
    if not raw["accepted"] or not occupancy["timing_eligible"] or occupancy["exit_code"] != 0:
        raise ValueError("failed or contended run")
    args = raw["arguments"]
    if args["repeats"] < 3 or len(raw["rows"]) != args["repeats"]:
        raise ValueError("at least three complete measured trials required")
    gpu = args["backend"] != "arpack"
    graph = raw["graph"]
    device_only = None
    if gpu:
        if graph is None or "HOST" in graph["node_types_including_conditional_body"]:
            raise ValueError("missing audit or graph host callback")
        if any(k.startswith("DtoH") for k in graph["memcpy_kinds"]):
            raise ValueError("graph readback")
        device_only = not any(k.startswith("HtoD") for k in graph["memcpy_kinds"])
        if args["backend"] in ("cg", "block") and not device_only:
            raise ValueError("host transfer in pure-Warp graph")
    elif graph is not None:
        raise ValueError("unexpected CPU graph audit")
    if raw["graph_device_only"] != device_only:
        raise ValueError("incorrect graph summary")
    for row in [raw["warmup"], *raw["rows"]]:
        q = row["quality"]
        if not q["accepted"] or len(row["values"]) != args["modes"]:
            raise ValueError("numerical failure or incomplete spectrum")
        if any(not math.isfinite(v) or v <= 0 for v in row["values"]):
            raise ValueError("invalid dominant values")
        if gpu and (q["status"] != 0 or q["converged"] != args["modes"]):
            raise ValueError("native failure")
        for key in ("residual", "orthogonality", "spectrum_error", "requested_subspace_error"):
            if not math.isfinite(q[key]) or not 0 <= q[key] < 1e-8:
                raise ValueError("failed original-pencil accuracy")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("invalid timing")
        geometry = row["geometry"]
        if args["modes"] == 1:
            if geometry is None or not geometry["valid_zero_set"] or geometry["crossings"] <= 0:
                raise ValueError("missing or invalid reconstructed zero set")
            for key in (
                "sample_distance_rms",
                "zero_set_relative_radial_rms",
                "zero_set_relative_radial_max",
                "normal_alignment_mean",
            ):
                if not math.isfinite(geometry[key]) or geometry[key] < 0:
                    raise ValueError("invalid geometry quality")
        elif geometry is not None:
            raise ValueError("individual clustered vectors do not define a unique surface")
        if args["backend"] == "cg":
            for key in ("cg_stats", "fresh_cg_stats"):
                stats = row[key]
                if len(stats) != 4 or any(v < 0 for v in stats) or not 0 <= stats[2] <= stats[3]:
                    raise ValueError("invalid inner diagnostics")
            if row["cg_stats"][3] != row["fresh_cg_stats"][3] or row["cg_stats"][3] == 0:
                raise ValueError("incomplete fresh inner audit")
            if len(row["fresh_residual_maxima"]) != 2 or any(
                not math.isfinite(v) or v < 0 for v in row["fresh_residual_maxima"]
            ):
                raise ValueError("invalid fresh residual audit")
        if args["backend"] == "block":
            stats = row["preconditioner_stats"]
            if len(stats) != 4 or any(v < 0 for v in stats) or not 0 <= stats[2] <= stats[3]:
                raise ValueError("invalid truncated-preconditioner diagnostics")
    for key in (
        "assembly_seconds",
        "reference_and_copy_assembly_seconds",
        "setup_seconds",
        "capture_seconds",
    ):
        if not math.isfinite(raw[key]) or raw[key] < 0:
            raise ValueError("invalid construction cost")
    if raw["setup_seconds"] <= 0 or (raw["capture_seconds"] > 0) != gpu:
        raise ValueError("missing setup/capture accounting")
    result = dict(
        setup_seconds=raw["setup_seconds"],
        capture_seconds=raw["capture_seconds"],
        solve_seconds=statistics.median(r["seconds"] for r in raw["rows"]),
        setup_capture_solve_seconds=raw["setup_seconds"]
        + raw["capture_seconds"]
        + raw["warmup"]["seconds"],
        residual=max(r["quality"]["residual"] for r in raw["rows"]),
        subspace_error=max(r["quality"]["requested_subspace_error"] for r in raw["rows"]),
        geometry=raw["rows"][0]["geometry"],
        graph_device_only=device_only,
        memory_bytes=raw["setup_capture_device_free_memory_decrease_bytes"],
        signature={
            k: raw[k] for k in ("matrix_sha256", "source_sha256", "environment", "versions", "n")
        },
    )
    result["signature"]["dependency"] = dependency_signature(raw["dependency"])
    result["signature"]["arguments"] = {k: v for k, v in args.items() if k != "out"}
    if args["backend"] == "cg":
        result["inner_diagnostics"] = dict(
            max_recursive_misses=max(r["cg_stats"][2] for r in raw["rows"]),
            max_fresh_misses=max(r["fresh_cg_stats"][2] for r in raw["rows"]),
            max_scheduled_iterations=max(r["cg_stats"][0] for r in raw["rows"]),
            max_fresh_relative_residual=max(r["fresh_residual_maxima"][1] for r in raw["rows"]),
        )
    if args["backend"] == "block":
        if raw["inner_role"] != "truncated preconditioner":
            raise ValueError("incorrect inner-solve contract")
        result["inner_diagnostics"] = dict(
            max_scheduled_iterations=max(r["preconditioner_stats"][0] for r in raw["rows"]),
            max_recursive_tolerance_misses=max(r["preconditioner_stats"][2] for r in raw["rows"]),
        )
    return result


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("finite regression limit >=1 required")
    if current["signature"] != baseline["signature"]:
        raise ValueError("incompatible baseline")
    ratio = current["solve_seconds"] / baseline["solve_seconds"]
    if not math.isfinite(ratio) or ratio > limit:
        raise ValueError(f"solve regression: {ratio}")
    return ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.report.read_text()),
        json.loads(args.report.with_suffix(".occupancy.json").read_text()),
    )
    if args.baseline:
        result["baseline_ratio"] = compare(result, json.loads(args.baseline.read_text()))
    args.report.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "signature"}, indent=2))


if __name__ == "__main__":
    main()
