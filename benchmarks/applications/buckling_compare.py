"""Signed tetrahedral buckling comparisons with original-pencil acceptance gates."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.acceptance import add_diagnostic_argument, assess_comparison, require_accepted
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import BucklingEigensolver, CGInverse, CuDSSInverse, SparseOperator

from .compare import other_gpu_processes, run, timed
from .prestress_buckling import make_buckling_case
from .references import cupy_setup
from .runner import accuracy


def block_matrix(a):
    a = a.tobsr(blocksize=(3, 3))
    return ws.bsr_from_triplets(
        a.shape[0] // 3,
        a.shape[1] // 3,
        wp.array(np.repeat(np.arange(a.shape[0] // 3), np.diff(a.indptr)), dtype=wp.int32),
        wp.array(a.indices, dtype=wp.int32),
        wp.array(a.data, dtype=wp.mat33d),
    )


def warp_setup(p, method, ncv, outer_tol, inner_tol, fsai_width):
    # The fixture independently certifies K-sigma*G SPD. This is not a rule
    # allowing CG at arbitrary interior buckling shifts.
    shifted = block_matrix(p.a - p.target * p.b)
    if method == "warp-cudss":
        inverse = CuDSSInverse(shifted, 1, mtype="spd", refinement_steps=1)
    else:
        pre = linear.FSAI(
            shifted,
            max_row_size=fsai_width,
            max_step_size=3,
            apply_lanes=8,
            factor_dtype=wp.float32,
        )
        inverse = CGInverse(shifted, tol=inner_tol, atol=0.0, maxiter=10000, preconditioner=pre)
    solver = BucklingEigensolver(
        SparseOperator(matrix(p.a)),
        SparseOperator(matrix(p.b)),
        inverse,
        p.k,
        p.target,
        ncv=ncv,
        tol=outer_tol,
        method="krylov",
    )
    return solver, inverse


def gpu_run(p, method, args, destination, label, sign, competing):
    """Retain partial trial evidence and an explicit row for native exceptions."""
    row = dict(method=method, accepted=False)
    try:
        _gpu_run(p, method, args, destination, label, sign, competing, row)
    except Exception as error:
        import traceback

        row.update(
            accepted=False, outcome="error", reason=str(error), traceback=traceback.format_exc()
        )
    return row


def _gpu_run(p, method, args, destination, label, sign, competing, row):
    inverse = solver = graph = None
    if method == "cupy-reciprocal":
        reciprocal = replace(p, api_class="C", a=sign * p.b, b=p.a, target=0.0, mode="regular")
        setup_seconds, (native, cleanup, _) = timed(
            lambda: cupy_setup(
                reciprocal,
                args.ncv,
                lobpcg_tol=args.cupy_tol,
                lobpcg_maxiter=args.cupy_steps,
            )
        )

        def solve():
            mu, v, _ = native()
            return sign / mu, v, None

        get = None
        capture_seconds, audit = 0.0, None
    else:
        setup_seconds, (solver, inverse) = timed(
            lambda: warp_setup(p, method, args.ncv, args.outer_tol, args.inner_tol, args.fsai_width)
        )
        cleanup = getattr(inverse, "close", lambda: None)
        capture_seconds, graph = timed(lambda: solver.capture(args.steps))

        def solve():
            if method == "warp-cg":
                inverse.reset_stats()
            wp.capture_launch(graph)

        def get():
            return solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, None

        dot = Path("build") / (destination.stem + "-" + label + "-" + method + ".dot")
        dot.parent.mkdir(exist_ok=True)
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        dot.unlink()
    try:
        warmup_seconds, _ = timed(solve)
        samples, checks = [], []
        row.update(
            samples_seconds=samples,
            validation=checks,
            setup_seconds=setup_seconds,
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            graph=audit,
        )
        for _ in range(args.repeats):
            competing.update(other_gpu_processes())
            seconds, output = timed(solve)
            competing.update(other_gpu_processes())
            w, v, _ = get() if get else tuple(x.get() if hasattr(x, "get") else x for x in output)
            q = accuracy(p, w, v)
            q["physical_accepted"] = q["accepted"]
            if solver is not None:
                q.update(
                    converged=int(solver.converged.numpy()[0]),
                    core_converged=int(solver.core.converged.numpy()[0]),
                    solver_status=int(solver.core.status.numpy()[0]),
                )
                q["outer_stopping_accepted"] = (
                    q["converged"] == p.k and q["core_converged"] == p.k and q["solver_status"] == 0
                )
                q["accepted"] &= q["outer_stopping_accepted"]
                if method == "warp-cg":
                    q["cg_stats"] = inverse.stats.numpy().tolist()
                    q["inner_solve_accepted"] = q["cg_stats"][2] == 0
                    q["accepted"] &= q["inner_solve_accepted"]
            else:
                absolute = np.linalg.norm(p.b @ v - (p.a @ v) * (1 / w), axis=0)
                q["native_absolute_residuals"] = absolute.tolist()
                q["native_stopping_passed"] = bool(absolute.max() <= args.cupy_tol)
                q["accepted"] &= q["native_stopping_passed"]
            samples.append(seconds)
            checks.append(q)
        row.update(
            method=method,
            setup_seconds=setup_seconds,
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            samples_seconds=samples,
            solve_seconds=float(np.median(samples)),
            validation=checks,
            accepted=all(q["accepted"] for q in checks),
            graph=audit,
        )
    finally:
        if solver is not None:
            solver.core.graph = solver.graph = None
        graph = None
        cleanup()


def assess_baseline(report, baseline):
    """Compare matching signed rows with unchanged input/budget/runtime contracts."""
    ignored = {"methods", "signs", "out", "baseline", "repeats", "diagnostic"}
    same = {k: v for k, v in report["args"].items() if k not in ignored} == {
        k: v for k, v in baseline["args"].items() if k not in ignored
    }
    same &= all(
        report.get(key) == baseline.get(key)
        for key in ("gpu", "versions", "blas_threads", "upstream_source_sha256")
    )
    same &= all(
        data.get("numerical_contract_accepted", False) and data.get("timing_eligible", False)
        for data in (report, baseline)
    )
    old = {(row["sign"], row["method"]): row for row in baseline["cases"]}
    for row in report["cases"]:
        previous = old.get((row["sign"], row["method"]), {})
        comparable = same and row.get("accepted", False) and previous.get("accepted", False)
        comparable &= all(
            row.get(key) == previous.get(key) for key in ("matrix_sha256", "n", "k", "sigma")
        )
        row["baseline_comparable"] = bool(comparable)
        if comparable:
            row["time_ratio"] = row["solve_seconds"] / previous["solve_seconds"]
            row["performance_regression"] = row["time_ratio"] > 1.25
    report["baseline_comparable"] = bool(report["cases"]) and all(
        row["baseline_comparable"] for row in report["cases"]
    )
    report["performance_regression"] = any(
        row.get("performance_regression", False) for row in report["cases"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=6)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--signs", default="positive,negative")
    parser.add_argument("--shift", type=float, default=0.01)
    parser.add_argument("--methods", default="scipy,spectra,warp-cudss,warp-cg,cupy-reciprocal")
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--outer-tol", type=float, default=1e-9)
    parser.add_argument("--inner-tol", type=float, default=1e-14)
    parser.add_argument("--fsai-width", type=int, default=32)
    parser.add_argument("--cupy-steps", type=int, default=400)
    parser.add_argument("--cupy-tol", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline")
    add_diagnostic_argument(parser)
    args = parser.parse_args()
    methods, signs = args.methods.split(","), args.signs.split(",")
    if set(methods) - {"scipy", "spectra", "warp-cudss", "warp-cg", "cupy-reciprocal"} or set(
        signs
    ) - {"positive", "negative"}:
        parser.error("unknown method or sign")
    if min(args.repeats, args.steps, args.ncv, args.fsai_width, args.cupy_steps) < 1:
        parser.error("budgets must be positive")
    if any(not np.isfinite(x) or x <= 0 for x in [args.outer_tol, args.inner_tol, args.cupy_tol]):
        parser.error("tolerances must be finite and positive")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        args=vars(args),
        gpu=wp.get_device().name,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        spectra_binary_sha256=hashlib.sha256(
            Path("build/application_spectra.so").read_bytes()
        ).hexdigest()
        if Path("build/application_spectra.so").exists()
        else None,
        upstream_source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(linear.__file__), Path(linear.__file__).with_name("fsai.py")]
            if f.exists()
        },
        versions={
            n: importlib.metadata.version(n)
            for n in ["warp-lang", "numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted(Path("warpack").glob("*.py")),
                Path(__file__),
                Path("benchmarks/applications/prestress_buckling.py"),
                Path("benchmarks/applications/fem_cases.py"),
                Path("benchmarks/applications/runner.py"),
                Path("benchmarks/applications/references.py"),
                Path("benchmarks/acceptance.py"),
            ]
        },
        cases=[],
        protocol="Clamped P1 tetrahedral K SPD / G indefinite prescribed initial-stress pencil. Signed transformed buckling ordering, original equations, complete requested K-space and K-orthogonality. Independent reciprocal reference seed913; native seed2026, Warp seed42; no reference starts. Warm solve timings exclude fixture/reference/setup/capture. Warp physical and reciprocal-core counts plus upstream recursive CG convergence are mandatory; fresh inner CG diagnostics are tracked separately in issue89. Native CuPy has no buckling API: explicit reciprocal LOBPCG with SPD K metric and supplied cuDSS K inverse is labeled separately. Its fresh absolute reciprocal residual is mandatory.",
    )
    competing = set(other_gpu_processes())
    for label in signs:
        sign = 1 if label == "positive" else -1
        case = make_buckling_case(args.side, args.modes, sign, args.shift)
        p = case.problem
        if not p.k < args.ncv <= p.n:
            parser.error("ncv must exceed complete requested cluster and not exceed n")
        digest = hashlib.sha256()
        for a in [p.a, p.b]:
            for v in [a.indptr, a.indices, a.data]:
                digest.update(v.tobytes())
        for method in methods:
            print(label, method, "n", p.n, "k", p.k, flush=True)
            if method in ("scipy", "spectra"):
                row = run("G", p.n, method, args.repeats, args.ncv, args.steps, problem=p)
            else:
                row = gpu_run(p, method, args, destination, label, sign, competing)
            row.update(
                sign=label,
                n=p.n,
                k=p.k,
                nnz=p.a.nnz,
                geometric_nnz=p.b.nnz,
                matrix_sha256=digest.hexdigest(),
                reference=case.reference,
                reference_residual=case.reference_residual,
                reference_orthogonality=case.reference_orthogonality,
                transformed_selection_margin=case.selection_margin,
                shifted_spd=case.shifted_spd,
                sigma=p.target,
            )
            report["cases"].append(row)
            print(label, method, row["accepted"], row.get("solve_seconds"), flush=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
    report["competing_gpu_processes"] = sorted(competing)
    report["timing_eligible"] = not competing
    assess_comparison(report, report["cases"], args.diagnostic)
    if args.baseline:
        assess_baseline(report, json.loads(Path(args.baseline).read_text()))
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if competing:
        raise SystemExit("GPU contention; not timing evidence")
    require_accepted(report)
    if args.baseline and (not report["baseline_comparable"] or report["performance_regression"]):
        raise SystemExit("Buckling baseline is not comparable or exceeds the 1.25 regression gate")


if __name__ == "__main__":
    main()
