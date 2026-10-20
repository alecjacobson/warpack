"""Sparse damped/gyroscopic comparison with independent physical-mode gates.

Run under benchmarks.monitor_gpu. Each synchronized timed replay is validated;
setup, capture, warmup and reference construction are reported separately.
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
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.runtime import warp_runtime_hashes
from benchmarks.upstream.adapters import output

from .damped import coupled_chain, quality
from .damped_backends import initial, scipy_setup, spectra_setup, warp_setup
from .damped_pencil import consistent_chain


def accepted_physical(check, statistics, k):
    """The requested equation is authoritative; core counts remain diagnostics."""
    return bool(check["accepted"] and statistics["count"] == k and statistics.get("status", 0) == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--consistent-mass", action="store_true")
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument(
        "--kind",
        choices=["proportional", "nonproportional", "gyroscopic", "coupled-mass"],
        default="nonproportional",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["scipy", "spectra", "warp-cudss", "warp-gmres", "cupy"],
        default=["scipy", "spectra", "warp-cudss", "cupy"],
    )
    parser.add_argument("--reference-tol", type=float)
    parser.add_argument("--reference-refinement", type=int, default=0)
    parser.add_argument("--kinematic", action="store_true")
    parser.add_argument("--polish", type=int, default=0)
    parser.add_argument("--pencil-rayleigh", action="store_true")
    parser.add_argument("--balance", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--inner-restart", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.repeats, args.cycles, args.tol) <= 0
        or not args.k + 2 <= args.ncv <= 4 * args.nodes
    ):
        parser.error("require positive repeats/cycles/tolerance and k+2<=ncv<=4*nodes")
    if not np.isfinite(args.balance) or args.balance <= 0:
        parser.error("balance must be finite and positive")
    if args.reference_tol is not None and (
        not np.isfinite(args.reference_tol) or args.reference_tol <= 0
    ):
        parser.error("reference tolerance must be finite and positive")
    if args.polish < 0 or args.reference_refinement < 0:
        parser.error("polish must be nonnegative")
    if args.pencil_rayleigh and not args.consistent_mass:
        parser.error("pencil Rayleigh postprocessing requires --consistent-mass")
    if args.consistent_mass and (
        args.kind not in ("nonproportional", "gyroscopic") or args.balance != 1.0
    ):
        parser.error("consistent mass supports nonproportional/gyroscopic original coordinates")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    case = (consistent_chain if args.consistent_mass else coupled_chain)(
        args.nodes, args.k, args.kind
    )
    report = dict(
        protocol=__doc__,
        reference_setup_seconds=time.perf_counter() - begin,
        kind=args.kind,
        n=case.a.shape[0],
        nnz=case.a.nnz,
        b_nnz=None if case.b is None else case.b.nnz,
        generalized_pencil=case.b is not None,
        k=args.k,
        ncv=args.ncv,
        target=[case.target.real, case.target.imag],
        tol=args.tol,
        cycles=args.cycles,
        inner_restart=args.inner_restart,
        coordinate_scale=args.balance,
        inverse_polish_steps=args.polish,
        pencil_rayleigh=args.pencil_rayleigh,
        recover_velocity=args.kinematic,
        reference_refinement_steps=args.reference_refinement,
        reference_tol=args.tol if args.reference_tol is None else args.reference_tol,
        gpu=wp.get_device().name,
        warp_runtime=wp.__version__,
        start_sha256=hashlib.sha256(initial(case, args.balance).tobytes()).hexdigest(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        omp_threads=os.environ.get("OMP_NUM_THREADS"),
        repeats=args.repeats,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        matrix_sha256=hashlib.sha256(
            case.a.indptr.tobytes() + case.a.indices.tobytes() + case.a.data.tobytes()
        ).hexdigest(),
        mass_sha256=None
        if case.b is None
        else hashlib.sha256(
            case.b.indptr.tobytes() + case.b.indices.tobytes() + case.b.data.tobytes()
        ).hexdigest(),
        runtime_hashes=warp_runtime_hashes("build/damped_spectra.so"),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "warp-lang", "nvidia-cudss-cu12")
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *Path("warpack").glob("*.py"),
                Path("benchmarks/experimental/damped_coordinates.py"),
                Path("benchmarks/experimental/pencil_polish.py"),
                *Path("benchmarks/applications").glob("damped*.*"),
            ]
        },
        methods={},
    )
    for method in args.methods:
        row = {}
        report["methods"][method] = row

        def close():
            pass

        if method == "cupy":
            row.update(
                status="unsupported",
                reason="CuPy has no sparse general eigs; this matrix is not Hermitian and LOBPCG is not a substitute.",
            )
            continue
        try:
            begin = time.perf_counter()
            if method.startswith("warp-"):
                solver = warp_setup(
                    case,
                    args.ncv,
                    args.tol,
                    method[5:],
                    balance=args.balance,
                    polish=args.polish,
                    kinematic=args.kinematic,
                    inner_restart=args.inner_restart if method == "warp-gmres" else None,
                    pencil_rayleigh=args.pencil_rayleigh,
                )
                close = solver.close
                wp.synchronize()
                row["setup_seconds"] = time.perf_counter() - begin
                begin = time.perf_counter()
                graph = solver.capture(args.cycles)
                wp.synchronize()
                row["capture_seconds"] = time.perf_counter() - begin
                dot = args.out.with_suffix(".dot")
                wp.capture_debug_dot_print(graph, str(dot))
                row["graph"] = graph_summary(dot)
                dot.unlink()

                def run():
                    wp.capture_launch(graph)
                    wp.synchronize()

                def extract():
                    return (
                        output(solver.eigenvalues),
                        output(solver.eigenvectors).T,
                        dict(
                            count=int(solver.converged.numpy()[0]),
                            status=int(solver.status.numpy()[0]),
                            core_count=int(solver.core.converged.numpy()[0]),
                        ),
                    )
            else:
                solve, close = (scipy_setup if method == "scipy" else spectra_setup)(
                    case,
                    args.ncv,
                    args.tol if args.reference_tol is None else args.reference_tol,
                    balance=args.balance,
                    polish=args.polish,
                    reference_refinement=args.reference_refinement,
                    kinematic=args.kinematic,
                    pencil_rayleigh=args.pencil_rayleigh,
                )
                row["setup_seconds"] = time.perf_counter() - begin
                latest = None

                def run():
                    nonlocal latest
                    latest = solve()

                def extract():
                    return latest

            begin = time.perf_counter()
            run()
            row["warmup_seconds"] = time.perf_counter() - begin
            row["trials"] = []
            for _ in range(args.repeats):
                begin = time.perf_counter()
                run()
                elapsed = time.perf_counter() - begin
                values, vectors, stats = extract()
                check = quality(case, values, vectors)
                check["accepted"] = accepted_physical(check, stats, case.k)
                row["trials"].append(dict(seconds=elapsed, quality=check, statistics=stats))
            row["accepted"] = all(t["quality"]["accepted"] for t in row["trials"])
            row["median_seconds"] = float(np.median([t["seconds"] for t in row["trials"]]))
        except Exception as error:
            row.update(accepted=False, error=f"{type(error).__name__}: {error}")
        finally:
            close()
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(method, row, flush=True)
    report["accepted"] = all(
        row.get("accepted", row.get("status") == "unsupported")
        for row in report["methods"].values()
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
