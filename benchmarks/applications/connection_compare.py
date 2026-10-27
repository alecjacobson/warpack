"""Original-pencil gated native-complex versus realified connection solves."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.acceptance import add_diagnostic_argument, assess_comparison, require_accepted
from benchmarks.graph_audit import graph_summary
from benchmarks.runtime import complete_warp_fingerprint, warp_runtime_hashes
from benchmarks.upstream.adapters import output

from . import connection_backends as backends
from .compare import other_gpu_processes
from .connection import make_connection_case
from .runner import accuracy


def timed(fn):
    wp.synchronize()
    start = time.perf_counter()
    result = fn()
    wp.synchronize()
    return time.perf_counter() - start, result


def run(args):
    p = make_connection_case(args.side, args.modes, args.mass, args.copies).problem
    digest = hashlib.sha256()
    for matrix in (p.a, backends.mass(p)):
        for data in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(data.tobytes())
    row = dict(
        method=args.method,
        n=p.n,
        k=p.k,
        ncv=args.ncv,
        real_n=2 * p.n if args.method.startswith("warp") else None,
        real_ncv=2 * args.ncv if args.method.startswith("warp") else None,
        mass=args.mass,
        copies=args.copies,
        nnz=p.a.nnz,
        mass_nnz=backends.mass(p).nnz,
        shift=p.target,
        matrix_sha256=digest.hexdigest(),
        samples=[],
        validation=[],
        graph=None,
        outer_cycles=args.steps,
        seed=2026,
        cg_tol=args.cg_tol,
        correction_steps=args.correction_steps if args.method == "warp-cg-corrected" else None,
        correction_tol=args.correction_tol if args.method == "warp-cg-corrected" else None,
    )

    def close():
        return None

    solver = graph = inverse = None
    try:
        if args.method.startswith("warp"):
            row["setup_seconds"], (solver, inverse) = timed(
                lambda: backends.warp_setup(
                    p,
                    args.ncv,
                    args.method.removeprefix("warp-"),
                    args.cg_tol,
                    correction_steps=args.correction_steps,
                    correction_tol=args.correction_tol,
                )
            )
            close = getattr(inverse, "close", close)
            row["capture_seconds"], graph = timed(lambda: solver.capture(args.steps))

            def solve():
                if hasattr(inverse, "reset_stats"):
                    inverse.reset_stats()
                wp.capture_launch(graph)

            def get(_):
                return solver.eigenvalues.numpy(), output(solver.eigenvectors).T

            dot = Path(args.out).with_suffix(".dot")
            wp.capture_debug_dot_print(graph, str(dot))
            row["graph"] = graph_summary(dot)
            row["device_only_graph"] = row["graph"]["node_types_including_conditional_body"].get(
                "HOST", 0
            ) == 0 and all("DtoD" in kind for kind in row["graph"]["memcpy_kinds"])
        else:
            setup = {
                "scipy": backends.scipy_setup,
                "scipy-ritz": lambda p, ncv: backends.scipy_setup(p, ncv, recover_metric=True),
                "spectra": backends.spectra_setup,
                "cupy": lambda p, ncv: backends.cupy_setup(p),
                "cupy-corrected": lambda p, ncv: backends.corrected_cupy_setup(p),
                "cupy-eigsh": backends.cupy_eigsh_setup,
            }[args.method]
            row["setup_seconds"], (solve, close) = timed(lambda: setup(p, args.ncv))

            def get(result):
                return tuple(x.get() if hasattr(x, "get") else x for x in result[:2])

            row["capture_seconds"] = None
        row["warmup_seconds"], _ = timed(solve)
        contenders = other_gpu_processes()
        for _ in range(args.repeats):
            elapsed, result = timed(solve)
            w, v = get(result)
            check = accuracy(p, w, v)
            if solver is not None:
                check.update(
                    complex_count=int(solver.count.numpy()[0]),
                    physical_count=int(solver.converged.numpy()[0]),
                    real_count=int(solver.real.converged.numpy()[0]),
                    solver_status=int(solver.real.status.numpy()[0]),
                )
                check["solver_full_count"] = (
                    check["complex_count"] == p.k
                    and check["physical_count"] == p.k
                    and check["real_count"] == 2 * p.k
                )
                if hasattr(inverse, "stats"):
                    check["cg_stats"] = inverse.stats.numpy().tolist()
                    check["cg_fresh_stats"] = inverse.fresh_stats.numpy().tolist()
                    check["cg_residual_maxima"] = inverse.residual_maxima.numpy().tolist()
                    if args.method == "warp-cg-corrected":
                        check["initial_cg_stats"] = (
                            inverse.initial_stats.numpy().tolist()
                            if inverse.initial_stats is not None
                            else inverse.stats.numpy().tolist()
                        )
                        check["correction_cg_stats"] = (
                            inverse.refinement_stats.numpy().tolist()
                            if inverse.refinement_stats is not None
                            else [0, 0, 0, 0]
                        )
                    check["inner_solve_accepted"] = (
                        check["cg_stats"][2] == 0 and check["cg_fresh_stats"][2] == 0
                    )
            elif args.method in ("cupy", "cupy-corrected"):
                history = result[2]
                last = history[-1].get()
                absolute = np.linalg.norm(p.a @ v - (backends.mass(p) @ v) * w, axis=0)
                check.update(
                    native_iterations=len(history),
                    native_history_max=float(np.max(last)),
                    native_fresh_max=float(np.max(absolute)),
                )
                check["native_stopping_passed"] = bool(
                    np.max(last) <= 1e-9 and np.max(absolute) <= 1e-9
                )
            row["samples"].append(elapsed)
            row["validation"].append(check)
        contenders.extend(other_gpu_processes())
        row["timing_uncontended"] = not contenders
        row["competing_processes"] = contenders
        row["solve_seconds"] = float(np.median(row["samples"]))
        row["accepted"] = all(v["accepted"] for v in row["validation"])
        row["inner_solve_accepted"] = all(
            v.get("inner_solve_accepted", True) for v in row["validation"]
        )
        row["outcome"] = "passed" if row["accepted"] else "failed"
    except Exception as error:
        import traceback

        row.update(
            accepted=False, outcome="error", error=repr(error), traceback=traceback.format_exc()
        )
    finally:
        if solver is not None:
            solver.real.graph = None
        graph = None
        close()
    return row


def assess_baseline(report, baseline):
    runtime = ("gpu", "warp", "versions", "blas_threads", "runtime_hashes")
    same_runtime = all(
        report.get(key) is not None and report.get(key) == baseline.get(key) for key in runtime
    )
    same_runtime &= all(
        item.get("numerical_contract_accepted", False) for item in (report, baseline)
    )
    same_runtime &= all(
        complete_warp_fingerprint(item.get("runtime_hashes")) for item in (report, baseline)
    )
    old = {row["method"]: row for row in baseline["cases"]}
    keys = (
        "matrix_sha256",
        "n",
        "k",
        "ncv",
        "real_ncv",
        "outer_cycles",
        "cg_tol",
        "seed",
        "correction_steps",
        "correction_tol",
    )
    for row in report["cases"]:
        previous = old.get(row["method"], {})
        comparable = same_runtime and all(row.get(key) == previous.get(key) for key in keys)
        comparable &= all(
            item.get("numerical_contract_accepted", False) and item.get("timing_uncontended", False)
            for item in (row, previous)
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
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument(
        "--mass", choices=["identity", "lumped", "consistent"], default="consistent"
    )
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="scipy,spectra,cupy,warp-cudss,warp-cg-fsai")
    parser.add_argument("--method", default="warp-cudss")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--cg-tol", type=float, default=1e-13)
    parser.add_argument("--correction-steps", type=int, default=1)
    parser.add_argument("--correction-tol", type=float, default=0.01)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    add_diagnostic_argument(parser)
    args = parser.parse_args()
    if (
        not 1 <= args.modes < args.ncv <= args.side**2 * args.copies
        or min(args.steps, args.repeats) < 1
    ):
        parser.error("require 1<=modes<ncv<=n, positive steps and repetitions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        args.out.write_text(json.dumps(run(args), indent=2) + "\n")
        return
    methods = args.methods.split(",")
    if set(methods) - {
        "scipy",
        "scipy-ritz",
        "spectra",
        "cupy",
        "cupy-corrected",
        "cupy-eigsh",
        "warp-cudss",
        "warp-cg",
        "warp-cg-fsai",
        "warp-cg-corrected",
    }:
        parser.error("unknown method")
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        implementation_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in list(Path(__file__).parent.glob("connection*.*"))
            + [Path("benchmarks/experimental/corrected_cg.py"), Path("warpack/generalized.py")]
            if path.suffix in (".py", ".cpp")
        },
        runtime_hashes=warp_runtime_hashes("build/connection_spectra.so"),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        fixture="Periodic flat P1 U(1) connection, independent Fourier and triangle references",
        acceptance=dict(residual=1e-8, orthogonality=1e-8, spectrum=1e-8, subspace=1e-8),
        cases=[],
    )
    for method in methods:
        destination = args.out.with_name(args.out.stem + "-" + method + ".json")
        command = [
            sys.executable,
            "-m",
            __spec__.name,
            "--worker",
            "--method",
            method,
            "--out",
            str(destination),
        ]
        for name in (
            "side",
            "modes",
            "mass",
            "copies",
            "ncv",
            "steps",
            "repeats",
            "cg_tol",
            "correction_steps",
            "correction_tol",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        with destination.with_suffix(".log").open("w") as log:
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                )
                if process.returncode:
                    raise RuntimeError(f"worker exit {process.returncode}")
                row = json.loads(destination.read_text())
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                row = dict(method=method, accepted=False, outcome="error", error=str(error))
        report["cases"].append(row)
        print(method, row.get("outcome"), row.get("solve_seconds"), flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    assess_comparison(report, report["cases"], diagnostic=args.diagnostic)
    if args.baseline:
        assess_baseline(report, json.loads(args.baseline.read_text()))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    require_accepted(report)
    if args.baseline and (not report["baseline_comparable"] or report["performance_regression"]):
        raise SystemExit("Baseline incomparable or 1.25 performance gate failed")


if __name__ == "__main__":
    main()
