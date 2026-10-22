"""Strict paired replay summaries for the wide FEM block comparison.

Setup stages retain their measured scope: the CuPy adapter reuses the existing
FSAI/base CG object, so its setup must not masquerade as standalone setup.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks.dependency_provenance import signature

FIELDS = (
    "residual",
    "orthogonality",
    "spectrum_error",
    "requested_subspace_error",
    "nullspace_subspace_error",
)


def summarize(raw, occupancy):
    if (
        not raw["accepted"]
        or not raw["timing_eligible"]
        or not occupancy["timing_eligible"]
        or occupancy["exit_code"] != 0
    ):
        raise ValueError("failed or contended worker")
    args, fixture = raw["args"], raw["fixture"]
    if not args["public_api"] or not args["cupy"] or args["repeats"] < 3:
        raise ValueError("public paired comparison needs three repeats")
    if (
        not fixture["boundary_cluster_complete"]
        or raw["k"] != fixture["actual_nontrivial_modes"] + fixture["retained_rigid_modes"]
    ):
        raise ValueError("incomplete requested cluster")
    if (
        fixture["requested_nontrivial_modes"] != args["modes"]
        or fixture["actual_nontrivial_modes"] < args["modes"]
    ):
        raise ValueError("missing requested modes")
    graph = raw["graph"]
    nodes = graph["node_types_including_conditional_body"]
    if (
        not nodes.get("KERNEL")
        or nodes.get("HOST", 0)
        or any("DtoD" not in kind for kind in graph["memcpy_kinds"])
    ):
        raise ValueError("non-device graph")
    adapter = raw["cupy_adapter"]
    groups = {"warp": [], "cupy": []}
    for row in raw["rows"]:
        if row["method"].startswith("Warp BlockLOBPCG"):
            name = "warp"
        elif row["method"].startswith("CuPy LOBPCG"):
            name = "cupy"
        else:
            raise ValueError("unknown method")
        q = row["quality"]
        if (
            not all(
                row[key]
                for key in ("accepted", "physical_and_outer_accepted", "native_stopping_accepted")
            )
            or not q["accepted"]
        ):
            raise ValueError("rejected solve")
        if any(not math.isfinite(q[f]) or not 0 <= q[f] < 1e-8 for f in FIELDS):
            raise ValueError("physical accuracy failure")
        if q["recovered_nullity"] != fixture["retained_rigid_modes"]:
            raise ValueError("wrong rigid mode count")
        if name == "warp":
            if q["status"] != 0 or q["converged"] != raw["k"] or q["residual"] > args["tol"]:
                raise ValueError("Warp native stopping failure")
        else:
            residuals = row["returned_absolute_residuals"]
            if (
                not row["full_block_absolute_tolerance_met"]
                or not row["returned_k_absolute_tolerance_met"]
                or len(residuals) != adapter["width"]
                or any(
                    not math.isfinite(x) or not 0 <= x <= adapter["absolute_tolerance"]
                    for x in residuals
                )
            ):
                raise ValueError("CuPy native stopping failure")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("invalid runtime")
        groups[name].append(row)
    if any(len(rows) != args["repeats"] for rows in groups.values()):
        raise ValueError("missing method repeats")
    setup = {key: raw[key] for key in ("total_setup_seconds", "capture_seconds", "warmup_seconds")}
    setup["cupy_adapter_and_upload_seconds"] = adapter["reference_setup_seconds"]
    if any(not math.isfinite(x) or x <= 0 for x in setup.values()):
        raise ValueError("invalid setup")
    return dict(
        arguments=args,
        n=raw["n"],
        k=raw["k"],
        fixture=fixture,
        setup_stages=setup,
        setup_scope=raw["setup_scope"],
        cupy_setup_scope=adapter["reference_setup_scope"],
        native_stopping=adapter["stopping"],
        dependency=raw["dependency"],
        versions=raw["versions"],
        source_sha256=raw["source_sha256"],
        methods={
            name: dict(
                median_seconds=statistics.median(r["seconds"] for r in rows),
                seconds=[r["seconds"] for r in rows],
                worst_quality={f: max(r["quality"][f] for r in rows) for f in FIELDS},
            )
            for name, rows in groups.items()
        },
        warmup_validation="The worker records a warmup duration but no warmup physical checks.",
    )


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("invalid regression limit")
    if signature(current["dependency"]) != signature(baseline["dependency"]):
        raise ValueError("incompatible dependency")
    for key in ("environment", "source_sha256", "versions", "fixture", "n", "k"):
        if current[key] != baseline[key]:
            raise ValueError("incompatible " + key)
    if {k: v for k, v in current["arguments"].items() if k != "out"} != {
        k: v for k, v in baseline["arguments"].items() if k != "out"
    }:
        raise ValueError("incompatible configuration")
    ratios = {
        name: current["methods"][name]["median_seconds"]
        / baseline["methods"][name]["median_seconds"]
        for name in ("warp", "cupy")
    }
    if any(not math.isfinite(x) or x > limit for x in ratios.values()):
        raise ValueError("replay regression")
    return ratios


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--baseline", type=Path)
    args = p.parse_args()
    report = summarize(
        json.loads(args.report.read_text()),
        json.loads(args.report.with_suffix(".occupancy.json").read_text()),
    )
    report["environment"] = json.loads((args.report.parent / "environment.json").read_text())
    if args.baseline:
        report["baseline_ratios"] = compare(report, json.loads(args.baseline.read_text()))
    args.report.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["methods"])


if __name__ == "__main__":
    main()
