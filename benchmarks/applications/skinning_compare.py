"""Accuracy-gated constrained skinning comparisons; reference code only."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.acceptance import add_diagnostic_argument, assess_comparison, require_accepted
from benchmarks.cudss_provenance import cudss_dependencies
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import ConstrainedKrylovSchur, CuDSSInverse, SparseOperator, saddle_point_matrix

from .compare import other_gpu_processes
from .skinning_cases import make_skinning_case, reference, validate


def spectra_setup(case, k, ncv, start):
    path = Path("build/skinning_spectra.so")
    source = Path(__file__).with_name("skinning_spectra.cpp")
    if not path.exists() or path.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            [
                "g++",
                "-O3",
                "-DNDEBUG",
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-I/usr/include/eigen3",
                "-Ibuild/upstream-spectra/include",
                str(source),
                "-o",
                str(path),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(path.resolve()))
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.skinning_factor.argtypes = [ctypes.c_int] * 3 + [ip, ip, dp, dp]
    lib.skinning_factor.restype = ctypes.c_void_p
    lib.skinning_solve.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        dp,
        ctypes.c_double,
        dp,
        dp,
    ]
    lib.skinning_solve.restype = ctypes.c_int
    lib.skinning_free.argtypes = [ctypes.c_void_p]
    a = case.kkt()
    begin = time.perf_counter()
    context = lib.skinning_factor(
        case.n, a.shape[0], a.nnz, a.indptr, a.indices, a.data, case.m.diagonal()
    )
    elapsed = time.perf_counter() - begin
    if not context:
        raise RuntimeError("Spectra KKT factorization failed")

    def solve():
        values, vectors = np.empty(k), np.empty(case.n * k)
        count = lib.skinning_solve(context, k, ncv, start, case.shift, values, vectors)
        if count != k:
            raise RuntimeError(f"Spectra converged {count}/{k}")
        return values, vectors.reshape((case.n, k), order="F"), None

    return solve, lambda: lib.skinning_free(context), elapsed


def run(case, method, k, ncv, steps, repeats, expected, basis, folder):
    n, r = case.n, case.c.shape[0]
    ncv = min(ncv, n - 1)
    scale = np.sqrt(case.m.diagonal())
    start = np.random.default_rng(2026).normal(size=n)
    row = dict(
        method=method,
        n=n,
        constraints=r,
        nnz=case.h.nnz,
        kkt_nnz=case.kkt().nnz,
        k=k,
        ncv=ncv,
        shift=case.shift,
        seed=2026,
        tolerance=1e-9 if method == "warp-cudss" else 1e-11,
        acceptance_tolerance=1e-7,
        steps=steps if method == "warp-cudss" else None,
    )

    def cleanup():
        pass

    solver = graph = None
    wp.synchronize()
    begin = time.perf_counter()
    if method == "scipy":
        factor = sla.splu(case.kkt())

        def product(x):
            return scale * factor.solve(np.r_[scale * x, np.zeros(r)])[:n]

        operator = sla.LinearOperator((n, n), matvec=product, dtype=np.float64)

        def solve():
            w, v = sla.eigsh(
                operator, k=k, ncv=ncv, which="LM", tol=1e-11, maxiter=10000, v0=start.copy()
            )
            return case.shift + 1 / w, v / scale[:, None], None

    elif method == "spectra":
        solve, cleanup, elapsed = spectra_setup(case, k, ncv, start)
    elif method in ("warp-cudss", "cupy-cudss"):
        h, m, c = matrix(case.h), matrix(case.m), matrix(case.c)
        inverse = CuDSSInverse(
            saddle_point_matrix(h, m, c, case.shift),
            1,
            mtype="general",
            matching=True,
            refinement_steps=1,
            cooperative_refinement=True,
        )
        cleanup = inverse.close
        row["factor"] = "cuDSS general + matching + one Warp residual correction"
        row["cooperative_refinement"] = True
        if method == "warp-cudss":
            solver = ConstrainedKrylovSchur(
                SparseOperator(h),
                SparseOperator(m),
                c,
                inverse,
                k,
                ncv=ncv,
                target=case.shift,
                tol=1e-9,
                initial_vector=wp.array(start / scale, dtype=wp.float64),
            )

            def solve():
                wp.capture_launch(graph)

        else:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as cl

            stream = cp.cuda.Stream.from_external(wp.get_stream())
            with stream:
                scaling = cp.asarray(scale)
                initial = cp.asarray(start)
                rhs = wp.zeros((1, n + r), dtype=wp.float64)
                output = wp.empty_like(rhs)
                rhs_cp, output_cp = cp.asarray(rhs), cp.asarray(output)

                def product(x):
                    with stream:
                        rhs_cp.fill(0)
                        rhs_cp[0, :n] = scaling * x
                        inverse.apply(rhs, output)
                        return scaling * output_cp[0, :n]

                operator = cl.LinearOperator((n, n), matvec=product, dtype=np.float64)

                def solve():
                    with stream:
                        cp.random.seed(2026)
                        w, v = cl.eigsh(
                            operator,
                            k=k,
                            ncv=ncv,
                            which="LM",
                            tol=1e-11,
                            maxiter=10000,
                            v0=initial.copy(),
                        )
                        return case.shift + 1 / w, v / scaling[:, None], None

            row["note"] = (
                "CuPy eigsh on explicit mass-scaled KKT inverse; cuDSS supplied by our adapter, NOT native CuPy factorization"
            )
    else:
        return dict(
            row,
            outcome="unsupported",
            accepted=False,
            reason="CG is not valid on the indefinite KKT matrix; a constrained iterative inverse is separate work (#49)",
        )
    wp.synchronize()
    row["setup_seconds"] = elapsed if method == "spectra" else time.perf_counter() - begin

    def host_result(result):
        if method == "warp-cudss":
            return (
                solver.eigenvalues.numpy(),
                solver.eigenvectors.numpy().T,
                solver.multipliers.numpy().T,
            )
        return tuple(x.get() if hasattr(x, "get") else x for x in result)

    try:
        begin = time.perf_counter()
        if solver is not None:
            graph = solver.capture(steps)
            dot = folder / f"skinning-{n}.dot"
            wp.capture_debug_dot_print(graph, str(dot))
            row["graph_audit"] = graph_summary(dot)
        warm = solve()
        wp.synchronize()
        row["capture_warmup_seconds"] = time.perf_counter() - begin
        w, v, mu = host_result(warm)
        checks = [validate(case, w, v, expected, basis, mu)]
        samples = []
        for _ in range(repeats):
            wp.synchronize()
            begin = time.perf_counter()
            result = solve()
            wp.synchronize()
            samples.append(time.perf_counter() - begin)
            w, v, mu = host_result(result)
            checks.append(validate(case, w, v, expected, basis, mu))
        row.update(
            samples=samples,
            solve_seconds=float(np.median(samples)),
            checks=checks,
            accepted=all(c["accepted"] for c in checks),
            cudss_dependency=cudss_dependencies()
            if method in ("warp-cudss", "cupy-cudss")
            else None,
        )
        row["outcome"] = "passed" if row["accepted"] else "failed"
        if solver is not None:
            row.update(
                strict_converged=int(solver.converged.numpy()[0]),
                device_equilibrium_residual=float(solver.residuals.numpy().max()),
                device_constraint_residual=float(solver.constraint_residuals.numpy().max()),
            )
    finally:
        if solver is not None:
            solver.graph = graph = None
        cleanup()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="scipy,spectra,cupy-cudss,warp-cudss,warp-cg")
    parser.add_argument("--out", default="results/skinning_comparisons.json")
    parser.add_argument("--baseline")
    add_diagnostic_argument(parser)
    args = parser.parse_args()
    methods = args.methods.split(",")
    if set(methods) - {"scipy", "spectra", "cupy-cudss", "warp-cudss", "warp-cg"}:
        parser.error("unknown method")
    if min(args.side, args.modes, args.steps, args.repeats) < 1 or args.ncv <= args.modes:
        parser.error("positive side/modes/steps/repeats and ncv > modes required")
    folder = Path("build/skinning-audit")
    folder.mkdir(parents=True, exist_ok=True)
    case = make_skinning_case(args.side)
    if args.modes >= case.n - case.c.shape[0]:
        parser.error("modes must be smaller than the finite constrained dimension")
    digest = hashlib.sha256()
    for a in (case.h, case.m, case.c):
        for data in (a.indptr, a.indices, a.data):
            digest.update(data.tobytes())
    fingerprint = digest.hexdigest()
    cache = folder / f"reference-{fingerprint[:16]}-shift{case.shift}-k{args.modes}.npz"
    print(f"Reference for n={case.n}, k={args.modes}; cache={cache.exists()}", flush=True)
    if cache.exists():
        saved = np.load(cache)
        expected, basis, reference_label = (
            saved["values"],
            saved["vectors"],
            str(saved["reference"]),
        )
    else:
        expected, basis, reference_label = reference(case, args.modes)
        np.savez(cache, values=expected, vectors=basis, reference=reference_label)
    reference_check = validate(case, expected, basis, expected, basis)
    if not reference_check["accepted"]:
        raise RuntimeError(reference_check)
    sources = hashlib.sha256()
    for path in sorted(
        list(Path("warpack").glob("*.py")) + list(Path(__file__).parent.glob("skinning*.*"))
    ):
        sources.update(str(path).encode())
        sources.update(path.read_bytes())
    report = dict(
        side=args.side,
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        matrix_hash=fingerprint,
        numerical_source_hash=sources.hexdigest(),
        reference_cache=str(cache),
        cpu_model=next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        cpu_contention="Not isolated; this driver does not sample CPU contention",
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        other_gpu_processes=other_gpu_processes(),
        reference=reference_label,
        reference_check=reference_check,
        raw_constraint_rows=len(case.raw_constraints),
        independent_constraints=case.c.shape[0],
        protocol="Shared heterogeneous tet weight stiffness, lumped mass, rank-reduced Appendix B.1 constraints; sequential methods; one warmup and repeated synchronized solves, all independently checked. Assembly, reference and validation excluded. GPU setup includes uploads/KKT assembly/factors; CPU setup includes KKT factors. Spectra compilation excluded. Fixed Warp replay includes physical residuals and multipliers; CPU/CuPy physical checks are outside solve. No speedups from rejected trials.",
        cases=[],
    )
    destination = Path(args.out)
    for method in methods:
        print("Starting", method, flush=True)
        row = run(
            case, method, args.modes, args.ncv, args.steps, args.repeats, expected, basis, folder
        )
        report["cases"].append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(
            method, row["outcome"], row.get("setup_seconds"), row.get("solve_seconds"), flush=True
        )
    assess_comparison(report, report["cases"], args.diagnostic)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if args.baseline:
        prior = json.loads(Path(args.baseline).read_text())
        for key in ("gpu", "warp", "versions", "matrix_hash", "blas_threads"):
            if prior[key] != report[key]:
                raise ValueError(f"baseline mismatch: {key}")
        checks = []
        for row in report["cases"]:
            old = next(
                (p for p in prior["cases"] if p["method"] == row["method"] and p["accepted"]), None
            )
            if old is not None:
                if any(row[key] != old[key] for key in ("k", "ncv", "steps", "tolerance", "shift")):
                    raise ValueError("baseline solver configuration mismatch")
                ratio = row["solve_seconds"] / old["solve_seconds"]
                checks.append(
                    dict(
                        method=row["method"],
                        ratio=ratio,
                        accuracy_regression=not row["accepted"],
                        performance_regression=ratio > 1.25,
                    )
                )
        report["regression_checks"] = checks
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if not checks or any(
            c["accuracy_regression"] or c["performance_regression"] for c in checks
        ):
            raise SystemExit("Regression detected; evidence preserved")
    require_accepted(report)


if __name__ == "__main__":
    main()
