"""Accuracy-gated indefinite FEM comparisons with upstream Warp GMRES and cuDSS."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from warp.optim import linear

from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.applications.interior_fem import NAMES, make_interior_case
from benchmarks.applications.negative_foundation import make_negative_foundation
from benchmarks.applications.runner import accuracy, warp_output
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, GeneralizedKrylovSchur, GMRESInverse, SparseOperator


def gmres_inverse(
    shifted,
    positive,
    *,
    restart,
    maxiter,
    inner_tol,
    preconditioner,
    fsai_width=64,
    adaptive_capture=True,
):
    """Use upstream GMRES with a fixed preconditioner of a separate SPD proxy.

    ``positive`` is K0+M for the foundation fixture. K0 is the diffusion
    stiffness before subtracting the foundation; the shifted solve stays
    indefinite. No SPD factorization is applied to that shifted matrix.
    """
    pre = None
    if preconditioner not in ("none", "diag", "fsai"):
        raise ValueError("unknown GMRES preconditioner")
    if preconditioner != "none":
        pre = (
            linear.FSAI(positive, max_row_size=fsai_width, max_step_size=3, factor_dtype=wp.float64)
            if preconditioner == "fsai"
            else linear.preconditioner(positive, ptype="diag")
        )
    return GMRESInverse(
        shifted,
        tol=inner_tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        preconditioner=pre,
        adaptive_capture=adaptive_capture,
    )


def assess_trial(problem, solver, inverse, method):
    """Keep physical, outer and fresh inner acceptance distinct and mandatory."""
    q = accuracy(problem, *warp_output(problem, solver))
    q.update(
        physical_accepted=q["accepted"],
        status=int(solver.status.numpy()[0]),
        converged=int(solver.converged.numpy()[0]),
    )
    q["outer_stopping_accepted"] = q["status"] == 0 and q["converged"] == problem.k
    q["inner_solve_accepted"] = method != "gmres" or int(inverse.stats.numpy()[2]) == 0
    q["accepted"] = (
        q["physical_accepted"] and q["outer_stopping_accepted"] and q["inner_solve_accepted"]
    )
    return q


def compare_baseline(report, baseline):
    """Compare only accepted trials with the same inputs, budgets and runtime."""
    ignored = {"out", "baseline", "repeats"}
    same = {k: v for k, v in report["args"].items() if k not in ignored} == {
        k: v for k, v in baseline["args"].items() if k not in ignored
    }
    same &= all(
        report[k] == baseline[k]
        for k in (
            "matrix_sha256",
            "gpu",
            "warp",
            "versions",
            "blas_threads",
            "upstream_source_sha256",
        )
    )
    report["baseline_comparable"] = bool(
        same and report["timing_eligible"] and baseline["timing_eligible"]
    )
    old = {row["method"]: row for row in baseline["rows"]}
    for row in report["rows"]:
        prior = old.get(row["method"], {})
        row["baseline_comparable"] = bool(
            report["baseline_comparable"] and row["accepted"] and prior.get("accepted")
        )
        if row["baseline_comparable"]:
            row["time_ratio"] = row["median_seconds"] / prior["median_seconds"]
            row["performance_regression"] = row["time_ratio"] > 1.25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[*NAMES, "negative_lumped", "negative_consistent"],
        default="clamped_consistent",
    )
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--band-index", type=int, default=6)
    parser.add_argument("--fraction", type=float, default=0.37)
    parser.add_argument("--ncv", type=int, default=48)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--restart", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=1024)
    parser.add_argument("--inner-tol", type=float, default=1e-12)
    parser.add_argument("--outer-tol", type=float, default=1e-9)
    parser.add_argument("--adaptive-capture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline")
    parser.add_argument("--preconditioner", choices=["none", "diag", "fsai"], default="fsai")
    parser.add_argument("--fsai-width", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="gmres,cudss")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    methods = args.methods.split(",")
    if (
        set(methods) - {"gmres", "cudss"}
        or min(args.repeats, args.steps, args.restart, args.maxiter, args.fsai_width) < 1
        or any(not math.isfinite(t) or t <= 0 for t in (args.inner_tol, args.outer_tol))
    ):
        parser.error("expected gmres/cudss methods and positive repeats")
    before = other_gpu_processes()
    negative = args.case.startswith("negative_")
    case = (
        make_negative_foundation(
            args.side, args.modes, "SM", consistent=args.case == "negative_consistent"
        )
        if negative
        else make_interior_case(
            args.case, args.side, args.modes, band_index=args.band_index, fraction=args.fraction
        )
    )
    p = case.problem
    print("Prepared", args.case, "n=", p.n, "k=", p.k, flush=True)
    a, mass, shifted = matrix(p.a), matrix(p.b), matrix(p.a - p.target * p.b)
    if args.ncv <= p.k:
        parser.error("ncv must exceed the complete requested cluster")
    positive = matrix(p.a + (case.foundation + 1 if negative else 1) * p.b)
    digest = hashlib.sha256()
    for mat in (p.a, p.b):
        for values in (mat.indptr, mat.indices, mat.data):
            digest.update(values.tobytes())
    report = dict(
        args=vars(args),
        n=p.n,
        k=p.k,
        sigma=p.target,
        reference="Analytic complete tensor-product Q1 sine eigenspace"
        if negative
        else case.reference,
        matrix_sha256=digest.hexdigest(),
        nnz=p.a.nnz,
        mass_nnz=p.b.nnz,
        ncv=min(p.n, args.ncv),
        matrix_properties="Real symmetric indefinite shifted stiffness; SPD mass",
        preconditioner_matrix="SPD K0+M, before subtracting foundation" if negative else "SPD K+M",
        start="Warp seed 42 restored each replay; no reference vectors as starts",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "nvidia-cudss-cu12")
        },
        upstream_source_sha256={
            "linear": hashlib.sha256(Path(inspect.getfile(linear.gmres)).read_bytes()).hexdigest(),
            "fsai": hashlib.sha256(Path(inspect.getfile(linear.FSAI)).read_bytes()).hexdigest()
            if hasattr(linear, "FSAI")
            else None,
        },
        expected=p.expected.tolist(),
        selection_margin=None if negative else case.selection_margin,
        gpu=wp.get_device().name,
        warp=wp.__version__,
        rows=[],
        other_gpu_processes_before=before,
        source_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [
                __file__,
                "benchmarks/applications/negative_foundation.py",
                "benchmarks/applications/runner.py",
                *map(str, sorted(Path("warpack").glob("*.py"))),
            ]
        },
        protocol="Sequential warm replay trials; setup includes preconditioner/inverse and solver allocation, capture separate. CPU reference and device matrix transfer excluded. Every original-pencil and complete requested-space gate retained. GMRES uses a fixed right preconditioner of the stated SPD proxy, never an SPD factorization of K-sigma M. Actual inner iterations and fresh linear failures retained. cuDSS vendor HtoD metadata is tracked separately in #29.",
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:

        def setup():
            if method == "gmres":
                inverse = gmres_inverse(
                    shifted,
                    positive,
                    restart=args.restart,
                    maxiter=args.maxiter,
                    inner_tol=args.inner_tol,
                    preconditioner=args.preconditioner,
                    fsai_width=args.fsai_width,
                    adaptive_capture=args.adaptive_capture,
                )
            else:
                inverse = CuDSSInverse(shifted, 1, mtype="general", matching=True)
            solver = GeneralizedKrylovSchur(
                SparseOperator(a),
                SparseOperator(mass),
                inverse,
                p.k,
                ncv=min(p.n, args.ncv),
                target=p.target,
                tol=args.outer_tol,
            )
            return inverse, solver

        free_before = wp.get_device().free_memory
        setup_seconds, (inverse, solver) = timed(setup)
        free_after_setup = wp.get_device().free_memory
        print(method, "setup", setup_seconds, flush=True)
        capture_seconds, graph = timed(lambda: solver.capture(args.steps))
        print(method, "capture", capture_seconds, flush=True)
        memory = free_before - wp.get_device().free_memory
        dot = Path("build") / f"interior-{method}.dot"
        dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        dot.unlink()
        warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
        row = dict(
            method=method,
            setup_seconds=setup_seconds,
            setup_free_memory_decrease_bytes=free_before - free_after_setup,
            capture_free_memory_decrease_bytes=memory - (free_before - free_after_setup),
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            free_memory_decrease_bytes=memory,
            memory_scope="Setup/capture pool and runtime free-memory decrease, not peak/live allocation",
            graph=audit,
            trials=[],
        )
        report["rows"].append(row)
        for _ in range(args.repeats):
            if method == "gmres":
                inverse.reset_stats()
            seconds, _ = timed(lambda: wp.capture_launch(graph))
            q = assess_trial(p, solver, inverse, method)
            trial = dict(
                seconds=seconds, quality=q, outer_iterations=int(solver.iterations.numpy()[0])
            )
            if method == "gmres":
                trial.update(
                    inner_stats=inverse.stats.numpy().tolist(),
                    inner_fresh_residual_maxima=inverse.residual_maxima.numpy().tolist(),
                )
            row["trials"].append(trial)
            print(method, seconds, q, flush=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
        row["accepted"] = all(t["quality"]["accepted"] for t in row["trials"])
        row["host_nodes"] = audit["node_types_including_conditional_body"].get("HOST", 0)
        row["device_only_graph"] = row["host_nodes"] == 0 and not any(
            "Host" in kind or "HOST" in kind for kind in audit["memcpy_kinds"]
        )
        row["graph_accepted"] = (
            row["host_nodes"] == 0
            and not any(kind.startswith("DtoH") for kind in audit["memcpy_kinds"])
            if method == "cudss"
            else row["device_only_graph"]
        )
        row["accepted"] &= row["graph_accepted"]
        row["median_seconds"] = float(np.median([r["seconds"] for r in row["trials"]]))
        solver.graph = graph = None
        if method == "cudss":
            inverse.close()
    report["other_gpu_processes_after"] = other_gpu_processes()
    report["timing_eligible"] = not (before or report["other_gpu_processes_after"])
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        compare_baseline(report, baseline)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["accepted"] for row in report["rows"]) or (
        args.baseline
        and not all(
            row.get("baseline_comparable") and not row.get("performance_regression")
            for row in report["rows"]
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
