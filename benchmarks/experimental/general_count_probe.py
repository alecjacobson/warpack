"""Accuracy diagnostics for the two tight-count Spectra fixtures in #35.

Requires prepare_spectra's original ComplexEigs/GenEigsRealShift fixtures.
Includes the first graph replay; this is not an accepted timing comparison.
Rayleigh/product diagnostics and dense reference work stay outside the solver.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.audit_profile import graph_summary
from benchmarks.runtime import warp_runtime_hashes
from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.run_spectra import FIX, read
from warpack import GeneralEigensolver, ShiftInvertEigensolver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=["ComplexEigs_16", "GenEigsRealShift_12"])
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--iterations", type=int)
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    if args.repeats < 1 or (args.iterations is not None and args.iterations < 1):
        p.error("require positive repeats and iterations")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    c = next(
        json.loads(line)
        for path in FIX.glob("*.jsonl")
        for line in path.read_text().splitlines()
        if json.loads(line)["id"] == args.case
    )
    a = read(FIX / (c["id"] + ".a"))
    ref = la.eigvals(a)
    shifted = c["family"] == "GenEigsRealShift"
    target = ref[np.argsort(abs(ref - c["sigma_real"]) if shifted else ref.real)[: c["k"]]]
    s = (
        ShiftInvertEigensolver(
            complex_matrix(a), c["k"], c["sigma_real"], ncv=c["ncv"], tol=args.tol
        )
        if shifted
        else GeneralEigensolver(complex_matrix(a), c["k"], ncv=c["ncv"], which="SR", tol=args.tol)
    )
    core = getattr(s, "core", s)
    budget = args.iterations or (30 if shifted else 300)
    trace_buffers = None
    if args.trace:
        from benchmarks.experimental.general_trace import capture_trace

        g, trace_buffers = capture_trace(s, budget)
    else:
        g = s.capture(budget)
    report = dict(
        actual_core_class=type(core).__module__ + "." + type(core).__qualname__,
        case=c,
        tol=args.tol,
        budget=args.iterations or (30 if shifted else 300),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        matrix_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        projected_sha256=hashlib.sha256(Path("warpack/schur.py").read_bytes()).hexdigest(),
        arnoldi_sha256=hashlib.sha256(Path("warpack/arnoldi.py").read_bytes()).hexdigest(),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        runtime_hashes=warp_runtime_hashes("build/application_spectra.so"),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "nvidia-cudss-cu12")
        },
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        warmup=args.warmup,
        timing_note="Synchronized replay after one warmup; setup, capture and validation excluded"
        if args.warmup
        else "Includes first replay; accuracy diagnostic, not matched accepted solve timing",
        trace_enabled=args.trace,
        replays=[],
    )
    dot = Path(args.out).with_suffix(".dot")
    wp.capture_debug_dot_print(g, str(dot))
    report["graph"] = graph_summary(dot)
    dot.unlink()
    if args.warmup:
        wp.capture_launch(g)
        wp.synchronize()
    for i in range(args.repeats):
        if trace_buffers is not None:
            trace_buffers[0].fill_(float("nan"))
        wp.synchronize()
        t = time.perf_counter()
        wp.capture_launch(g)
        wp.synchronize()
        elapsed = time.perf_counter() - t
        w, v = output(s.eigenvalues), output(s.eigenvectors).T
        av = a @ v
        rq = np.sum(v.conj() * av, axis=0) / np.sum(abs(v) ** 2, axis=0)
        original = np.linalg.norm(av - v * w, axis=0) / np.maximum(1, abs(w))
        optimal = np.linalg.norm(av - v * rq, axis=0) / np.maximum(1, abs(rq))
        ax = output(s.ax).T
        cost = abs(w[:, None] - target)
        ii, jj = linear_sum_assignment(cost)
        row = dict(
            replay=i,
            seconds=elapsed,
            count=int(s.converged.numpy()[0]),
            status=int(s.status.numpy()[0]),
            iterations=int(getattr(s, "core", s).iterations.numpy()[0]),
            device_residuals=s.residuals.numpy().tolist(),
            original_residuals=original.tolist(),
            rayleigh_residuals=optimal.tolist(),
            max_absolute_residual=float(abs(av - v * w).max()),
            norm_error=float(abs(np.linalg.norm(v, axis=0) - 1).max()),
            max_requested_spectrum_error=float(cost[ii, jj].max()),
            rayleigh_value_changes=abs(rq - w).tolist(),
            device_product_errors=np.linalg.norm(ax - av, axis=0).tolist(),
        )
        row["residual_recoveries"] = (
            int(getattr(s, "core", s).residual_restarts.numpy()[0])
            if hasattr(getattr(s, "core", s), "residual_restarts")
            else 0
        )
        if hasattr(core, "residual_fallbacks"):
            row["residual_fallbacks"] = int(core.residual_fallbacks.numpy()[0])
        row["core_count"] = int(core.converged.numpy()[0])
        row["core_residuals"] = core.residuals.numpy().tolist()
        row["accepted"] = bool(
            row["count"] == c["k"]
            and row["core_count"] == c["k"]
            and row["status"] == 0
            and row["max_absolute_residual"] <= (1e-8 if shifted else 1e-9)
            and row["max_requested_spectrum_error"] <= 1e-8
            and row["norm_error"] <= 1e-10
        )
        if trace_buffers is not None:
            trace, h, q = trace_buffers
            projected, ritz = output(core.projected), output(core.ritz)
            row["trace"] = trace.numpy().tolist()
            row["projected_residuals"] = (
                np.linalg.norm(output(h) @ projected - projected * ritz, axis=0)
                / np.maximum(1, abs(ritz))
            ).tolist()
            basis = output(q)
            row["pre_restart_orthogonality"] = float(
                np.linalg.norm(basis.conj() @ basis.T - np.eye(core.m))
            )
        report["replays"].append(row)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)
    if args.profile:
        report["profile_note"] = (
            "Instrumented eager restart after final replay, including CUDA event overhead"
        )
        with wp.ScopedTimer("general_restart", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
            core.step()
            core.finalize()
        grouped = {}
        for result in timer.timing_results:
            grouped[result.name] = grouped.get(result.name, 0.0) + result.elapsed
        report["instrumented_kernel_ms"] = grouped
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    report["accepted"] = all(row["accepted"] for row in report["replays"])
    report["solve_seconds"] = float(np.median([row["seconds"] for row in report["replays"]]))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if shifted:
        s.core.graph = s.graph = g = None
        s.close()
    if not report["accepted"] and not args.diagnostic:
        raise SystemExit("Requested spectrum, residual, norm or full-count gate failed")


if __name__ == "__main__":
    main()
