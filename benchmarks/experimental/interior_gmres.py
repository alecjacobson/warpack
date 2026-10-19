"""Accuracy-gated indefinite FEM comparisons with upstream Warp GMRES and cuDSS."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from warp.optim import linear

from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.applications.interior_fem import NAMES, make_interior_case
from benchmarks.applications.runner import accuracy, warp_output
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, GeneralizedKrylovSchur, GMRESInverse, SparseOperator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=NAMES, default="clamped_consistent")
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--band-index", type=int, default=6)
    parser.add_argument("--fraction", type=float, default=0.37)
    parser.add_argument("--ncv", type=int, default=48)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--restart", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=1024)
    parser.add_argument("--inner-tol", type=float, default=1e-12)
    parser.add_argument("--preconditioner", choices=["none", "diag", "fsai"], default="fsai")
    parser.add_argument("--fsai-width", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="gmres,cudss")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    methods = args.methods.split(",")
    if set(methods) - {"gmres", "cudss"} or args.repeats < 1:
        parser.error("expected gmres/cudss methods and positive repeats")
    before = other_gpu_processes()
    case = make_interior_case(
        args.case, args.side, args.modes, band_index=args.band_index, fraction=args.fraction
    )
    p = case.problem
    print("Prepared", args.case, "n=", p.n, "k=", p.k, flush=True)
    a, mass, shifted = matrix(p.a), matrix(p.b), matrix(p.a - p.target * p.b)
    report = dict(
        args=vars(args),
        n=p.n,
        k=p.k,
        sigma=p.target,
        reference=case.reference,
        expected=p.expected.tolist(),
        selection_margin=case.selection_margin,
        gpu=wp.get_device().name,
        warp=wp.__version__,
        rows=[],
        other_gpu_processes_before=before,
        source_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [__file__, "warpack/iterative.py", "warpack/generalized_krylov.py"]
        },
        protocol="Sequential warm replay trials; setup includes preconditioner/inverse and solver allocation, capture separate. CPU reference and device matrix transfer excluded. Every original-pencil and complete requested-space gate retained. GMRES uses a fixed right preconditioner of SPD K+M, never an SPD factorization of K-sigma M. Actual inner iterations and fresh linear failures retained. cuDSS vendor HtoD metadata is tracked separately in #29.",
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:

        def setup():
            if method == "gmres":
                pre = None
                if args.preconditioner != "none":
                    positive = matrix(p.a + p.b)
                    pre = (
                        linear.FSAI(
                            positive,
                            max_row_size=args.fsai_width,
                            max_step_size=3,
                            factor_dtype=wp.float64,
                        )
                        if args.preconditioner == "fsai"
                        else linear.preconditioner(positive, ptype="diag")
                    )
                inverse = GMRESInverse(
                    shifted,
                    tol=args.inner_tol,
                    restart=args.restart,
                    maxiter=args.maxiter,
                    preconditioner=pre,
                    adaptive_capture=True,
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
                tol=1e-9,
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
            q = accuracy(p, *warp_output(p, solver))
            q.update(
                status=int(solver.status.numpy()[0]), converged=int(solver.converged.numpy()[0])
            )
            q["accepted"] &= q["status"] == 0 and q["converged"] == p.k
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
        row["median_seconds"] = float(np.median([r["seconds"] for r in row["trials"]]))
        solver.graph = graph = None
        if method == "cudss":
            inverse.close()
    report["other_gpu_processes_after"] = other_gpu_processes()
    report["timing_eligible"] = not (before or report["other_gpu_processes_after"])
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
