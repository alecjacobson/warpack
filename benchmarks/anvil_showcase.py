"""Refresh all README Anvil rows serially, with independent setup repetitions."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.anvil_inputs import signature as input_signature
from benchmarks.cudss_provenance import cudss_signature
from benchmarks.dependency_provenance import signature

METHODS = ("arpack", "spectra", "cupy-direct", "warp-direct", "cupy-block", "warp-block")


def command(method, repeats, output):
    if method in ("arpack", "spectra"):
        return [
            "benchmarks.anvil_factor_audit",
            "--backend",
            "pardiso",
            "--library",
            "build/anvil-factor-audit/pardiso_spectra.so",
            "--pardiso-refinement",
            "0",
            f"--{method}-repeats",
            str(repeats),
            "--out",
            str(output),
        ]
    if method.endswith("direct"):
        return [
            "benchmarks.anvil_direct",
            "--backend",
            method.split("-")[0],
            "--repeats",
            str(repeats),
            "--out",
            str(output),
        ]
    if method == "cupy-block":
        return [
            "benchmarks.experimental.anvil_cupy_setup",
            "--repeats",
            str(repeats),
            "--out",
            str(output),
        ]
    return [
        "benchmarks.experimental.compare_block",
        "--public-api",
        "--case",
        "anvil",
        "--width",
        "24",
        "--steps",
        "200",
        "--shift",
        "100",
        "--fsai-width",
        "32",
        "--fsai-layout",
        "columns",
        "--matrix-layout",
        "columns",
        "--repeats",
        str(repeats),
        "--tol",
        "1e-8",
        "--gram-chunk",
        "4096",
        "--symmetric-products",
        "--fused-updates",
        "--paired-rotation",
        "--reuse-x-image",
        "--batched-products",
        "--inner-cg",
        "25",
        "--batched-cg",
        "--active-cg-bucket-size",
        "4",
        "--inner-fp32",
        "--inner-tol",
        ".01",
        "--active-search-bucket-size",
        "4",
        "--active-search-keep-history",
        "--out",
        str(output),
    ]


def extract(method, data):
    if method in ("arpack", "spectra"):
        trials = data[method]["trials"]
        return (
            data["inverse_setup_seconds"],
            [r["seconds"] for r in trials],
            max(r["max_original_residual"] for r in trials),
            data["inverse_checks_accepted"] and data[method]["all_accepted"],
        )
    if method == "cupy-block":
        rows = data["validation_rows"]
    else:
        rows = data["rows"]
    setup = (
        data["total_setup_seconds"] + data["capture_seconds"]
        if method == "warp-block"
        else data["setup_seconds"]
    )
    return (
        setup,
        [r["seconds"] for r in rows],
        max(r["quality"]["max_original_residual"] for r in rows),
        data["accepted"],
    )


def summarize(directory, rounds, repeats):
    rows = []
    dependency = None
    cudss_dependency = None
    common_inputs = None
    for method in METHODS:
        setups, solves, medians, residuals = [], [], [], []
        method_inputs = None
        for trial in range(1, rounds + 1):
            path = directory / f"{method}-{trial}.json"
            raw = json.loads(path.read_text())
            current_inputs = input_signature(raw.get("inputs"))
            shared = input_signature(current_inputs, common_only=True)
            if common_inputs is not None and common_inputs != shared:
                raise RuntimeError(f"Mixed Anvil matrices/mesh/reference: {path}")
            if method_inputs is not None and method_inputs != current_inputs:
                raise RuntimeError(f"Changed Anvil method inputs: {path}")
            common_inputs, method_inputs = shared, current_inputs
            current = signature(raw.get("dependency"))
            if dependency is not None and dependency != current:
                raise RuntimeError(f"Mixed dependency overlays: {path}")
            dependency = current
            if method.endswith("direct"):
                current_cudss = cudss_signature(raw.get("cudss_dependency"))
                if cudss_dependency is not None and cudss_dependency != current_cudss:
                    raise RuntimeError(f"Mixed loaded cuDSS libraries or bindings: {path}")
                cudss_dependency = current_cudss
            setup, samples, residual, accepted = extract(method, raw)
            occupancy = json.loads(path.with_suffix(".occupancy.json").read_text())
            if not accepted or not occupancy["timing_eligible"] or len(samples) != repeats:
                raise RuntimeError(f"Unqualified measurement: {path}")
            setups.append(setup)
            solves.extend(samples)
            medians.append(statistics.median(samples))
            residuals.append(residual)
        rows.append(
            dict(
                method=method,
                inputs=method_inputs,
                setup_median_seconds=statistics.median(setups),
                setup_samples_seconds=setups,
                solve_median_seconds=statistics.median(solves),
                solve_min_seconds=min(solves),
                solve_max_seconds=max(solves),
                solve_batch_medians_seconds=medians,
                solve_batch_median_spread_fraction=(max(medians) - min(medians))
                / statistics.median(medians),
                max_original_residual=max(residuals),
                timed_solves=len(solves),
                accepted=True,
            )
        )
    summary = dict(
        common_inputs=common_inputs,
        dependency=dependency,
        cudss_dependency=cudss_dependency,
        rows=rows,
        rounds=rounds,
        repeats=repeats,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MKL_THREADING_LAYER",
            )
        },
        protocol="Serial fresh-process setup repetitions; one outer solve warmup then synchronized solves from identical starts. Setup includes factor/preconditioner construction, solver workspaces and graph capture where applicable. Assembled inputs, imports/runtime initialization, reference loading, outer warmup and validation excluded. Existing compilation caches. CPU libraries allocate iteration workspaces inside their timed solve calls.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for trial in range(1, args.rounds + 1):
            for method in METHODS:
                output = args.out / f"{method}-{trial}.json"
                occupancy = output.with_suffix(".occupancy.json")
                if args.resume and output.exists() and occupancy.exists():
                    _, samples, _, accepted = extract(method, json.loads(output.read_text()))
                    if (
                        accepted
                        and len(samples) == args.repeats
                        and json.loads(occupancy.read_text())["timing_eligible"]
                    ):
                        continue
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.monitor_gpu",
                        "--out",
                        str(occupancy),
                        "--log",
                        f"build/anvil-{method}-{trial}.log",
                        "--",
                        sys.executable,
                        "-m",
                        *command(method, args.repeats, output),
                    ],
                    check=True,
                )
                if not extract(method, json.loads(output.read_text()))[3]:
                    raise RuntimeError(f"Accuracy failed: {output}")
    summarize(args.out, args.rounds, args.repeats)


if __name__ == "__main__":
    main()
