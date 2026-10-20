"""Bounded, isolated singular-pencil comparisons with original-equation gates."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.optim.linear as linear

from benchmarks.acceptance import add_diagnostic_argument, assess_comparison, require_accepted
from benchmarks.audit_profile import graph_summary

from . import conformal_backends as backends
from .compare import other_gpu_processes, timed
from .conformal import boundary_reference, make_conformal_case, metrics
from .conformal_warp import ConformalEigensolver
from .connection_compare import assess_baseline as connection_baseline

METHODS = (
    "scipy",
    "scipy-original",
    "spectra",
    "cupy",
    "cupy-scaled",
    "warp-cudss",
    "warp-cg",
    "warp-cg-fsai",
)


def run(args):
    case = make_conformal_case(args.side, args.modes, args.height, args.pin, args.shift)
    start = time.perf_counter()
    reference = boundary_reference(case)
    reference_seconds = time.perf_counter() - start
    digest = hashlib.sha256()
    for mat in case.reduced:
        for data in (mat.indptr, mat.indices, mat.data):
            digest.update(data.tobytes())
    digest.update(case.free.tobytes())
    digest.update(np.array([case.shift]).tobytes())
    row = dict(
        method=args.method,
        n=case.n,
        full_n=case.stiffness.shape[0],
        k=case.k,
        ncv=args.ncv,
        nnz=case.stiffness.nnz,
        mass_nnz=case.mass.nnz,
        finite_count=2 * (len(case.boundary) - 1),
        infinite_count=2 * (args.side - 2) ** 2,
        common_nullity_before_gauge=2,
        shift=case.shift,
        height=args.height,
        pin=args.pin,
        matrix_sha256=digest.hexdigest(),
        seed=2026,
        outer_cycles=args.steps,
        cg_tol=args.cg_tol,
        correction_steps=1,
        correction_tol=0.01,
        outer_tol=1e-10,
        cupy_tol=args.cupy_tol,
        cupy_maxiter=300,
        reciprocal_scale=-case.shift if args.method == "cupy-scaled" else 1.0,
        reference_seconds=reference_seconds,
        reference_values=reference[0].tolist(),
        reference="static boundary condensation + dense centered boundary eigensolve",
        samples=[],
        validation=[],
        graph=None,
    )
    solver = graph = None

    def close():
        return None

    try:
        if args.method.startswith("warp"):
            row["setup_seconds"], solver = timed(
                lambda: ConformalEigensolver(
                    case, args.ncv, args.method.removeprefix("warp-"), cg_tol=args.cg_tol
                )
            )
            close = solver.close
            row["capture_seconds"], graph = timed(lambda: solver.capture(args.steps))

            def solve():
                if hasattr(solver.inverse, "reset_stats"):
                    solver.inverse.reset_stats()
                wp.capture_launch(graph)

            def get(_):
                return solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T

            dot = args.out.with_suffix(".dot")
            wp.capture_debug_dot_print(graph, str(dot))
            row["graph"] = graph_summary(dot)
            row["device_only_graph"] = row["graph"]["node_types_including_conditional_body"].get(
                "HOST", 0
            ) == 0 and all("DtoD" in kind for kind in row["graph"]["memcpy_kinds"])
        else:
            setup = {
                "scipy": lambda: backends.scipy_setup(case, args.ncv),
                "scipy-original": lambda: backends.scipy_setup(case, args.ncv, True),
                "spectra": lambda: backends.spectra_reciprocal(case, args.ncv),
                "cupy": lambda: backends.cupy_reciprocal(case, tolerance=args.cupy_tol),
                "cupy-scaled": lambda: backends.cupy_reciprocal(
                    case, tolerance=args.cupy_tol, scaled=True
                ),
            }[args.method]
            row["setup_seconds"], (solve, close) = timed(setup)
            row["capture_seconds"] = None

            def get(result):
                return tuple(x.get() if hasattr(x, "get") else x for x in result[:2])

        row["warmup_seconds"], _ = timed(solve)
        contenders = other_gpu_processes()
        for _ in range(args.repeats):
            elapsed, result = timed(solve)
            w, v = get(result)
            check = metrics(case, w, v, reference)
            if solver is not None:
                check["physical_count"] = int(solver.converged.numpy()[0])
                check["reciprocal_count"] = int(solver.core.converged.numpy()[0])
                check["solver_status"] = int(solver.core.status.numpy()[0])
                check["solver_full_count"] = (
                    check["physical_count"] == case.k and check["reciprocal_count"] == case.k
                )
                if hasattr(solver.inverse, "stats"):
                    check["cg_stats"] = solver.inverse.stats.numpy().tolist()
                    check["cg_fresh_stats"] = solver.inverse.fresh_stats.numpy().tolist()
                    check["cg_residual_maxima"] = solver.inverse.residual_maxima.numpy().tolist()
                    check["inner_solve_accepted"] = (
                        check["cg_stats"][2] == 0 and check["cg_fresh_stats"][2] == 0
                    )
            if args.method in ("cupy", "cupy-scaled"):
                history, fresh = result[2]
                check["native_iterations"] = len(history)
                check["native_history_max"] = float(np.max(history[-1].get()))
                check["native_fresh_max"] = float(np.max(fresh.get()))
                check["native_stopping_passed"] = (
                    max(check["native_history_max"], check["native_fresh_max"]) <= row["cupy_tol"]
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
        graph = None
        close()
    return row


def assess_baseline(report, baseline):
    connection_baseline(report, baseline)
    old = {row["method"]: row for row in baseline["cases"]}
    for row in report["cases"]:
        if any(
            row.get(key) != old.get(row["method"], {}).get(key)
            for key in ("outer_tol", "cupy_tol", "cupy_maxiter", "reciprocal_scale")
        ):
            row["baseline_comparable"] = False
            row.pop("time_ratio", None)
            row.pop("performance_regression", None)
    report["baseline_comparable"] = bool(report["cases"]) and all(
        r["baseline_comparable"] for r in report["cases"]
    )
    report["performance_regression"] = any(
        r.get("performance_regression", False) for r in report["cases"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("side", 16),
        ("modes", 6),
        ("ncv", 32),
        ("steps", 8),
        ("repeats", 3),
        ("pin", 0),
        ("timeout", 240),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--height", type=float, default=0.25)
    parser.add_argument("--shift", type=float)
    parser.add_argument("--cg-tol", type=float, default=1e-11)
    parser.add_argument("--cupy-tol", type=float, default=1e-9)
    parser.add_argument(
        "--methods", default="scipy,scipy-original,spectra,cupy,warp-cudss,warp-cg-fsai"
    )
    parser.add_argument("--method", choices=METHODS, default="warp-cudss")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    add_diagnostic_argument(parser)
    args = parser.parse_args()
    if min(args.steps, args.repeats) < 1 or not args.modes < args.ncv <= 2 * args.side**2 - 2:
        parser.error("require positive repeats/steps and modes<ncv<=n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        args.out.write_text(json.dumps(run(args), indent=2) + "\n")
        return
    methods = args.methods.split(",")
    if set(methods) - set(METHODS):
        parser.error("unknown comparison method")
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12")
        },
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        implementation_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("conformal*.py")
        },
        runtime_hashes={
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if Path(path).exists()
            else None
            for name, path in {
                "warp_linear": inspect.getfile(linear),
                "spectra_binary": "build/application_spectra.so",
            }.items()
        },
        fixture="P1 conformal energy, centered boundary mass, translation quotient; no regularization",
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
            "ncv",
            "steps",
            "repeats",
            "pin",
            "height",
            "cg_tol",
            "cupy_tol",
            "shift",
        ):
            if getattr(args, name) is not None:
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
