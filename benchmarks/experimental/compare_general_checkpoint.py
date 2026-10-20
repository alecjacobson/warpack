"""Matched fixed-budget checkpoint overhead; every timed replay is validated."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.general_checkpoint import CheckpointGeneral
from benchmarks.runtime import warp_runtime_hashes
from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.check_general_references import read
from benchmarks.upstream.general_reference import GeneralReference
from benchmarks.upstream.run_benchmarks import StartedGeneral

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "build/spectra-benchmark-fixtures")
    parser.add_argument("--case", default="benchmark_gen_36")
    parser.add_argument("--ncv", type=int, default=60)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (args.fixtures / "benchmark_gen.jsonl").read_text().splitlines()
    ]
    case = next(case for case in cases if case["id"] == args.case)
    paths = [args.fixtures / (args.case + suffix) for suffix in (".a", ".v0")]
    a, initial = read(paths[0]), read(paths[1]).ravel()
    reference = GeneralReference(a, case["k"])
    wp.synchronize()
    start = time.perf_counter()
    op = complex_matrix(a)
    v0 = wp.array(np.column_stack([initial, np.zeros(len(a))]), dtype=wp.vec2d)
    wp.synchronize()
    common_setup = time.perf_counter() - start
    report = dict(
        case=case,
        n=len(a),
        nnz=int(np.count_nonzero(a)),
        k=case["k"],
        ncv=args.ncv,
        budget=args.budget,
        tol=1e-10,
        repeats=args.repeats,
        protocol="Matched fixed-budget solves on the same matrix/v0. Separate workspace "
        "and capture setup; warm synchronized reset replays alternate backend order. "
        "Original-equation, target spectrum/subspace, independence and count gates "
        "run outside every timed sample. Failed samples reject the comparison. "
        "Run through benchmarks.monitor_gpu for occupancy evidence. Clocks unlocked.",
        common_input_setup_seconds=common_setup,
        warp=wp.__version__,
        gpu=wp.get_device().name,
        runtime_hashes=warp_runtime_hashes(ROOT / "build/application_spectra.so"),
        matrix_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        start_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "warpack/arnoldi.py",
                "warpack/schur.py",
                "warpack/schur_tiled.py",
                "benchmarks/upstream/run_benchmarks.py",
                "benchmarks/upstream/general_reference.py",
                "benchmarks/experimental/general_checkpoint.py",
                "benchmarks/experimental/compare_general_checkpoint.py",
            )
        },
        variants={},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    solvers, graphs = {}, {}
    for name, cls in (("baseline", StartedGeneral), ("checkpoint", CheckpointGeneral)):
        start = time.perf_counter()
        solver = cls(op, case["k"], ncv=args.ncv, tol=1e-10)
        solver.start_vector = v0
        wp.synchronize()
        workspace_seconds = time.perf_counter() - start
        start = time.perf_counter()
        graph = solver.capture(args.budget, adaptive=False)
        wp.synchronize()
        capture_seconds = time.perf_counter() - start
        dot = args.out.with_name(args.out.stem + "-" + name + ".dot")
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        dot.unlink()
        solvers[name], graphs[name] = solver, graph
        report["variants"][name] = dict(
            workspace_setup_seconds=workspace_seconds,
            capture_seconds=capture_seconds,
            graph=audit,
            trials=[],
        )
    for graph in graphs.values():
        wp.capture_launch(graph)
        wp.synchronize()
    for repeat in range(args.repeats):
        order = ("baseline", "checkpoint") if repeat % 2 == 0 else ("checkpoint", "baseline")
        for name in order:
            solver, graph = solvers[name], graphs[name]
            wp.synchronize()
            start = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            elapsed = time.perf_counter() - start
            quality = reference.check(output(solver.eigenvalues), output(solver.eigenvectors).T)
            quality.update(
                converged=int(solver.converged.numpy()[0]), status=int(solver.status.numpy()[0])
            )
            quality["accepted"] &= quality["converged"] == case["k"] and quality["status"] == 0
            report["variants"][name]["trials"].append(dict(seconds=elapsed, quality=quality))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(repeat, name, elapsed, quality["accepted"], flush=True)
    accepted = True
    for variant in report["variants"].values():
        graph = variant["graph"]
        variant["accepted"] = all(trial["quality"]["accepted"] for trial in variant["trials"])
        variant["accepted"] &= graph["node_types_including_conditional_body"].get("HOST", 0) == 0
        variant["accepted"] &= all("DtoD" in kind for kind in graph["memcpy_kinds"])
        variant["median_seconds"] = float(
            np.median([trial["seconds"] for trial in variant["trials"]])
        )
        accepted &= variant["accepted"]
    report["accepted"] = bool(accepted)
    report["checkpoint_to_baseline_ratio"] = (
        (
            report["variants"]["checkpoint"]["median_seconds"]
            / report["variants"]["baseline"]["median_seconds"]
        )
        if accepted
        else None
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("accepted", accepted, "ratio", report["checkpoint_to_baseline_ratio"], flush=True)
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
