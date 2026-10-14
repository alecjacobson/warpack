"""Accuracy-gated application benchmarks. GPU methods run sequentially."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import warp as wp

from .problems import make_problem
from .references import Unsupported, cupy_setup, spectra_setup
from .runner import accuracy, reference_values, scipy_setup, warp_output, warp_setup


def timed(fn):
    wp.synchronize()
    begin = time.perf_counter()
    value = fn()
    wp.synchronize()
    return time.perf_counter() - begin, value


def other_gpu_processes():
    """Conservative contention check; not a substitute for a dedicated GPU."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
        )
        return [
            line.strip()
            for line in output.splitlines()
            if int(line.split(",")[0]) not in (os.getpid(), os.getppid())
        ]
    except (OSError, ValueError, subprocess.CalledProcessError):
        return ["GPU process inventory unavailable"]


def run(api_class, size, method, repeats=3, ncv=32, steps=None, problem=None):
    p = make_problem(api_class, size) if problem is None else problem
    ref = reference_values(p)
    matrix_hash = hashlib.sha256()
    for mat in (p.a, p.b):
        if mat is not None:
            csr = mat.tocsr()
            matrix_hash.update(csr.indptr.tobytes())
            matrix_hash.update(csr.indices.tobytes())
            matrix_hash.update(csr.data.tobytes())
    row = dict(
        api_class=api_class,
        requested_size=size,
        problem=p.name,
        n=p.n,
        columns=p.a.shape[1],
        nnz=p.a.nnz,
        mass_nnz=None if p.b is None else p.b.nnz,
        k=p.k,
        ncv=min(ncv, p.n),
        method=method,
        matrix_sha256=matrix_hash.hexdigest(),
        notes=p.notes,
        target=[float(np.real(p.target)), float(np.imag(p.target))],
        seed=2026,
    )

    def cleanup():
        return None

    solver = graph = None
    try:
        if method.startswith("warp"):
            selected = {"warp": "krylov", "warp-block": "block", "warp-cg": "cg"}[method]
            if method != "warp" and api_class not in "ABCD":
                raise Unsupported(
                    "Alternate Warp inverse/iteration comparison is limited to symmetric low/interior workloads"
                )
            if selected == "cg" and api_class == "B":
                raise Unsupported("Interior shifted matrix is not guaranteed SPD")
            if steps is None:
                steps = (
                    100
                    if selected == "block" or api_class in "DG"
                    else (5 if api_class in "ABC" else (10 if api_class in "EH" else 30))
                )
            row["outer_cycles"] = steps
            row["start"] = "Warp seed 42; same start restored on every replay"
            setup, (solver, inverse) = timed(lambda: warp_setup(p, selected, ncv))
            # Complex shift-invert wraps the real factor object twice.
            factor = getattr(inverse, "inverse", inverse)
            factor = getattr(factor, "inverse", factor)
            row["cudss_refinement_steps"] = getattr(factor, "refinement_steps", None)
            row["cudss_reordering"] = getattr(factor, "reordering", None)
            row["cudss_matching"] = getattr(factor, "matching", None)
            row["cudss_mtype"] = getattr(factor, "mtype", None)
            cleanup = getattr(inverse, "close", lambda: None)
            capture, graph = timed(lambda: solver.capture(steps))
            row.update(setup_seconds=setup, capture_seconds=capture)

            def solve():
                return wp.capture_launch(graph)

            def get():
                return warp_output(p, solver)

            audit = Path("build/issue-work") / f"graph-{api_class}-{size}-{method}.dot"
            audit.parent.mkdir(exist_ok=True, parents=True)
            wp.capture_debug_dot_print(graph, str(audit))
            dot = audit.read_text()
            row["graph_audit"] = dict(
                host_nodes=len(re.findall(r'label="\{\s*HOST\b', dot)),
                memcpy_kinds=dict(Counter(re.findall(r"\| \{kind \| ([^}]+)", dot))),
                method="CUDA graph DOT including nested bodies; not an API trace",
            )
        else:
            row["start"] = (
                "Reference seed 2026; restored per trial (algorithms have different start spaces)"
            )
            if method == "scipy":
                reference_ncv = min(ncv, min(p.a.shape) - 1 if p.mode == "svd" else p.n)
                row["reference_ncv"] = reference_ncv
                row["implementation"] = f"SciPy ARPACK explicit ncv={reference_ncv}"
                setup, solve = timed(lambda: scipy_setup(p, ncv=reference_ncv))
            elif method == "spectra":
                setup, (solve, cleanup) = timed(lambda: spectra_setup(p, ncv))
            elif method == "cupy":
                setup, (solve, cleanup, label) = timed(lambda: cupy_setup(p, ncv))
                row["implementation"] = label
                row["reference_tolerance"] = 1e-11 if api_class in "EI" else 1e-8
            else:
                raise ValueError(method)
            row["setup_seconds"] = setup
            get = None
        warmup, output = timed(solve)
        row["warmup_seconds"] = warmup
        samples = []
        checks = []
        competing = set()
        for _ in range(repeats):
            competing.update(other_gpu_processes())
            elapsed, output = timed(solve)
            competing.update(other_gpu_processes())
            host_output = (
                get()
                if get is not None
                else tuple(x.get() if hasattr(x, "get") else x for x in output)
            )
            check = accuracy(p, *host_output, reference=ref)
            samples.append(elapsed)
            checks.append(check)
        row.update(
            samples_seconds=samples,
            solve_seconds=float(np.median(samples)),
            validation=checks,
            accepted=all(c["accepted"] for c in checks),
            other_gpu_processes=sorted(competing),
            timing_uncontended=not competing,
        )
        if method.startswith("warp"):
            row["reported_converged"] = (
                int(solver.converged.numpy()[0]) if hasattr(solver, "converged") else None
            )
            row["status"] = int(solver.status.numpy()[0]) if hasattr(solver, "status") else None
            audit = row["graph_audit"]
            row["device_only_graph"] = audit["host_nodes"] == 0 and not any(
                "Host" in x or "HOST" in x for x in audit["memcpy_kinds"]
            )
        row["outcome"] = "passed" if row["accepted"] else "failed"
        row["contract_accepted"] = row["accepted"] and row.get("device_only_graph", True)
    except Unsupported as e:
        row.update(outcome="unsupported", reason=str(e), accepted=False)
    except Exception as e:
        import traceback

        row.update(outcome="error", reason=str(e), traceback=traceback.format_exc(), accepted=False)
    finally:
        if solver is not None and hasattr(solver, "graph"):
            solver.graph = None
        if solver is not None and hasattr(solver, "core"):
            solver.core.graph = None
        graph = None
        solver = None
        gc.collect()
        cleanup()
    return row


def compare_baseline(rows, baseline, limit):
    def key(row):
        return row["api_class"], row.get("requested_size", row.get("n")), row["method"]

    old = {key(r): r for r in baseline["cases"]}
    comparisons = []
    for r in rows:
        b = old.get(key(r))
        if b is None:
            continue
        same = all(
            r.get(key) == b.get(key)
            for key in ["matrix_sha256", "k", "ncv", "outer_cycles", "implementation"]
        )
        eligible = (
            same
            and b.get("accepted")
            and r.get("accepted")
            and b.get("timing_uncontended", False)
            and r.get("timing_uncontended", False)
        )
        item = dict(
            api_class=r["api_class"], n=r.get("n"), method=r["method"], comparable=bool(eligible)
        )
        if eligible:
            ratio = r["solve_seconds"] / b["solve_seconds"]
            item.update(time_ratio=ratio, performance_regression=ratio > limit)
        item["accuracy_regression"] = bool(b.get("accepted") and not r.get("accepted"))
        item["graph_regression"] = bool(
            b.get("device_only_graph") and not r.get("device_only_graph")
        )
        comparisons.append(item)
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", default="ABCDEFGHI")
    parser.add_argument("--sizes", default="80,1024")
    parser.add_argument("--methods", default="scipy,spectra,cupy,warp")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--out", default="results/application_coverage.json")
    parser.add_argument("--baseline")
    parser.add_argument("--regression-limit", type=float, default=1.25)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    dest = Path(args.out)
    dest.parent.mkdir(exist_ok=True, parents=True)
    if args.worker:
        row = run(args.classes, int(args.sizes), args.methods, args.repeats, args.ncv, args.steps)
        dest.write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n)
            for n in ["scipy", "numpy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        acceptance=dict(residual=1e-8, orthogonality=1e-8, spectrum=1e-8),
        protocol="Sequential isolated workers; setup and capture separate; synchronized wall median; every sample validated in original equations. GPU output conversion is outside timed solves. Acceptance is numerical; device_only_graph is reported separately. Start spaces differ by algorithm. Failed timings are diagnostics only.",
        cases=[],
    )
    build = Path("build/issue-work/comparisons")
    build.mkdir(exist_ok=True, parents=True)
    import sys

    for api_class in args.classes:
        for size in args.sizes.split(","):
            for method in args.methods.split(","):
                out = build / f"{api_class}-{size}-{method}.json"
                log = out.with_suffix(".log")
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.applications.compare",
                    "--worker",
                    "--classes",
                    api_class,
                    "--sizes",
                    size,
                    "--methods",
                    method,
                    "--repeats",
                    str(args.repeats),
                    "--ncv",
                    str(args.ncv),
                    "--out",
                    str(out),
                ]
                if args.steps is not None:
                    command += ["--steps", str(args.steps)]
                try:
                    with log.open("w") as f:
                        subprocess.run(
                            command, stdout=f, stderr=f, check=True, timeout=args.timeout
                        )
                    row = json.loads(out.read_text())
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    row = dict(
                        api_class=api_class,
                        requested_size=int(size),
                        method=method,
                        accepted=False,
                        outcome="timeout"
                        if isinstance(e, subprocess.TimeoutExpired)
                        else "worker_error",
                        log=str(log),
                    )
                report["cases"].append(row)
                dest.write_text(json.dumps(report, indent=2) + "\n")
                print(api_class, size, method, row["outcome"], row.get("solve_seconds"), flush=True)
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if baseline["gpu"] != report["gpu"] or baseline["warp"] != report["warp"]:
            raise ValueError("Baseline GPU/runtime mismatch")
        report["regression_checks"] = compare_baseline(
            report["cases"], baseline, args.regression_limit
        )
        report["performance_comparisons"] = sum(
            c["comparable"] for c in report["regression_checks"]
        )
        dest.write_text(json.dumps(report, indent=2) + "\n")
        if any(
            c.get("performance_regression") or c["accuracy_regression"] or c["graph_regression"]
            for c in report["regression_checks"]
        ):
            raise SystemExit(1)
        if not report["performance_comparisons"]:
            raise SystemExit(
                "No comparable uncontended timings; performance regression is inconclusive"
            )


if __name__ == "__main__":
    main()
