"""Accuracy-only explicit-width general solver profiles on Spectra's fixtures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.runtime import warp_runtime_hashes
from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.check_general_references import read
from benchmarks.upstream.general_reference import GeneralReference
from benchmarks.upstream.run_benchmarks import StartedGeneral

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "build/spectra-benchmark-fixtures")
    parser.add_argument("--case", action="append")
    parser.add_argument("--ncv-multiplier", type=int, default=2)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Experimental converged-set preservation"
    )
    parser.add_argument("--fixed", action="store_true", help="Execute every budgeted cycle")
    parser.add_argument("--ncv", type=int, help="Explicit width instead of the multiplier")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.ncv_multiplier, args.budget, args.repeats, args.ncv or 1) < 1:
        parser.error("width multiplier, budget and repeats must be positive")
    if args.checkpoint and not args.fixed:
        parser.error("The checkpoint experiment requires --fixed")
    cases = [
        json.loads(line)
        for line in (args.fixtures / "benchmark_gen.jsonl").read_text().splitlines()
    ]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unknown fixture case")
    solver_type = StartedGeneral
    if args.checkpoint:
        from benchmarks.experimental.general_checkpoint import CheckpointGeneral

        solver_type = CheckpointGeneral
    report = dict(
        protocol="Accuracy-only reset graph replays; public GeneralEigensolver with original "
        "Spectra v0, LM, tol=1e-10 and explicit ncv multiplier. No performance claims. "
        "Every replay must pass fresh original residuals, target assignment, "
        "mandatory modes, vector norms, independence, subspace and device counters.",
        warp=wp.__version__,
        gpu=wp.get_device().name,
        runtime_hashes=warp_runtime_hashes(ROOT / "build/application_spectra.so"),
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "warpack/arnoldi.py",
                "warpack/schur.py",
                "warpack/schur_tiled.py",
                "benchmarks/upstream/run_benchmarks.py",
                "benchmarks/upstream/general_reference.py",
                "benchmarks/experimental/general_selection_profiles.py",
                "benchmarks/experimental/general_checkpoint.py",
            )
        },
        budget=args.budget,
        fixed=args.fixed,
        checkpoint=args.checkpoint,
        requested_ncv=args.ncv,
        repeats=args.repeats,
        ncv_multiplier=args.ncv_multiplier,
        cases=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        matrix_path, initial_path = (
            args.fixtures / (case["id"] + suffix) for suffix in (".a", ".v0")
        )
        a, initial = read(matrix_path), read(initial_path).ravel()
        reference = GeneralReference(a, case["k"])
        ncv = min(len(a), args.ncv if args.ncv is not None else args.ncv_multiplier * case["ncv"])
        solver = solver_type(complex_matrix(a), case["k"], ncv=ncv, which="LM", tol=1e-10)
        solver.start_vector = wp.array(np.column_stack([initial, np.zeros(len(a))]), dtype=wp.vec2d)
        graph = solver.capture(args.budget, adaptive=not args.fixed)
        row = dict(
            case=case,
            ncv=ncv,
            keep=solver.keep,
            matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            start_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
            trials=[],
        )
        for _ in range(args.repeats):
            wp.capture_launch(graph)
            values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
            trial = reference.check(values, vectors)
            trial.update(
                converged=int(solver.converged.numpy()[0]),
                status=int(solver.status.numpy()[0]),
                iterations=int(solver.iterations.numpy()[0]),
                values=np.stack([values.real, values.imag], axis=-1).tolist(),
            )
            trial["accepted"] &= trial["converged"] == case["k"] and trial["status"] == 0
            if args.checkpoint:
                trial["checkpoint_updates"] = int(solver.checkpoint_updates.numpy()[0])
                trial["checkpoint_valid"] = int(solver.checkpoint_valid.numpy()[0])
            row["trials"].append(trial)
        row["accepted"] = all(trial["accepted"] for trial in row["trials"])
        report["cases"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            case["id"],
            ncv,
            "PASS" if row["accepted"] else "FAIL",
            [trial["iterations"] for trial in row["trials"]],
            flush=True,
        )
    report["summary"] = dict(
        passed=sum(row["accepted"] for row in report["cases"]), total=len(cases)
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
