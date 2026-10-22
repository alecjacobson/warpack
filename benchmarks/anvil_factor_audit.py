"""Profile CPU inverse setup on the exact mass-normalized Anvil pencil.

Run one worker at a time; set OPENBLAS_NUM_THREADS and OMP_NUM_THREADS before
startup. Optional ARPACK/Spectra runs also validate all 26 Anvil modes.
"""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from benchmarks import anvil_inputs
from benchmarks.dependency_provenance import warp_dependencies


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["assemble", "eigen", "superlu", "pardiso"], required=True)
    p.add_argument("--matrix", default=str(anvil_inputs.fixture_path().with_suffix(".shift1.npz")))
    p.add_argument("--library", default="build/anvil-factor-audit/eigen.so")
    p.add_argument("--arpack-repeats", type=int, default=0)
    p.add_argument("--spectra-repeats", type=int, default=0)
    p.add_argument("--arpack-tol", type=float, default=1e-12)
    p.add_argument("--pardiso-refinement", type=int, default=-1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.arpack_repeats < 0 or (args.backend == "assemble" and args.arpack_repeats):
        p.error("--arpack-repeats must be nonnegative and requires a factor backend")
    if args.spectra_repeats < 0 or (
        args.spectra_repeats and args.backend not in ("eigen", "pardiso")
    ):
        p.error("--spectra-repeats requires an Eigen/PARDISO library with AUDIT_SPECTRA")
    mesh = Path("anvil for export-ftetwild.mesh")
    arrays, manifest = anvil_inputs.load()
    original = anvil_inputs.host_matrix(arrays)
    expected = (original + sp.eye(original.shape[0], format="csr")).tocsc()
    expected.sort_indices()
    if args.backend == "assemble":
        matrix = expected
        sp.save_npz(args.matrix, matrix)
    else:
        matrix = sp.load_npz(args.matrix)
        if anvil_inputs.matrix_identity(matrix) != anvil_inputs.matrix_identity(expected):
            raise ValueError(
                "CPU matrix cache differs from verified Anvil fixture; rerun --backend assemble"
            )
    del expected, arrays
    input_identity = None
    if args.arpack_repeats or args.spectra_repeats:
        reference, reference_identity = anvil_inputs.reference()
        initial = np.load("build/anvil_initial.npy")
        input_identity = anvil_inputs.inputs(manifest, reference_identity, initial, matrix)
    if (
        matrix.format != "csc"
        or matrix.shape[0] != matrix.shape[1]
        or matrix.dtype != np.float64
        or matrix.indices.dtype != np.int32
        or matrix.indptr.dtype != np.int32
    ):
        raise ValueError("expected a square FP64 CSC matrix with int32 indices")
    report = {
        "inputs": input_identity,
        "fixture": manifest,
        "dependency": warp_dependencies(),
        "backend": args.backend,
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(
            matrix.indptr.tobytes() + matrix.indices.tobytes() + matrix.data.tobytes()
        ).hexdigest(),
        "n": matrix.shape[0],
        "nnz": matrix.nnz,
        "explicit_zeros": int(np.count_nonzero(matrix.data == 0)),
        "input_storage_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MKL_THREADING_LAYER",
            ]
        },
        "pid": os.getpid(),
        "arpack_tol": args.arpack_tol,
        "pardiso_refinement": args.pardiso_refinement,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "factor_library_sha256": (
            hashlib.sha256(Path(args.library).read_bytes()).hexdigest()
            if args.backend in ("eigen", "pardiso")
            else None
        ),
    }
    print(json.dumps(report), flush=True)
    start, cpu = time.perf_counter(), time.process_time()
    if args.backend == "superlu":
        lu = sla.splu(
            matrix,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        report.update(
            factor_seconds=time.perf_counter() - start, factor_cpu_seconds=time.process_time() - cpu
        )
        print(json.dumps(report), flush=True)
        report.update(l_nnz=lu.L.nnz, u_nnz=lu.U.nnz, factor_nnz=lu.L.nnz + lu.U.nnz)
        solve = lu.solve
    elif args.backend in ("eigen", "pardiso"):
        lib = ctypes.CDLL(str(Path(args.library).resolve()))
        if args.backend == "pardiso" and args.pardiso_refinement >= 0:
            lib.audit_configure.argtypes = [ctypes.c_int]
            lib.audit_configure(args.pardiso_refinement)
        ip = ctypes.POINTER(ctypes.c_int)
        dp = ctypes.POINTER(ctypes.c_double)
        lib.factor_audit.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp, dp]
        lib.factor_audit.restype = ctypes.c_void_p
        lib.audit_solve.argtypes = [ctypes.c_void_p, dp, dp]
        lib.audit_free.argtypes = [ctypes.c_void_p]
        stats = np.zeros(6)
        ctx = lib.factor_audit(
            matrix.shape[0],
            matrix.nnz,
            matrix.indptr.ctypes.data_as(ip),
            matrix.indices.ctypes.data_as(ip),
            matrix.data.ctypes.data_as(dp),
            stats.ctypes.data_as(dp),
        )
        if not ctx:
            raise RuntimeError(f"{args.backend} factorization failed")
        factor_fields = (
            ["strict_lower_nnz", "minimum_d"]
            if args.backend == "eigen"
            else ["pardiso_factor_nnz", "pardiso_factor_mflops"]
        )
        report.update(
            zip(
                [
                    "copy_seconds",
                    "symbolic_seconds",
                    "numeric_seconds",
                    "numeric_cpu_seconds",
                    *factor_fields,
                ],
                stats.tolist(),
            )
        )
        report["factor_seconds"] = float(sum(stats[:3]))
        report["factor_nnz"] = int(stats[4]) + (matrix.shape[0] if args.backend == "eigen" else 0)

        def solve(rhs):
            out = np.empty_like(rhs)
            lib.audit_solve(ctx, rhs.ctypes.data_as(dp), out.ctypes.data_as(dp))
            return out

    if args.backend != "assemble":
        report["inverse_setup_seconds"] = time.perf_counter() - start
        report["factor_nnz_over_full_input_nnz"] = report["factor_nnz"] / matrix.nnz
        rng = np.random.default_rng(731)
        report["solve_checks"] = []
        for _ in range(3):
            rhs = rng.standard_normal(matrix.shape[0])
            start = time.perf_counter()
            answer = solve(rhs)
            elapsed = time.perf_counter() - start
            residual = np.linalg.norm(matrix @ answer - rhs) / np.linalg.norm(rhs)
            report["solve_checks"].append({"seconds": elapsed, "relative_residual": residual})
        report["inverse_checks_accepted"] = all(
            row["relative_residual"] < 1e-7 for row in report["solve_checks"]
        )
        for method, repeats in [("arpack", args.arpack_repeats), ("spectra", args.spectra_repeats)]:
            if not repeats:
                continue
            from benchmarks.anvil import validate

            calls = [0]

            def matvec(rhs):
                calls[0] += 1
                return solve(rhs)

            op = sla.LinearOperator(matrix.shape, matvec=matvec, dtype=np.float64)
            if method == "spectra":
                lib.audit_spectra.argtypes = [ctypes.c_void_p, ctypes.c_double, dp, dp, dp, ip]
                lib.audit_spectra.restype = ctypes.c_int
                native_values = np.empty(26)
                native_vectors = np.empty((26, matrix.shape[0]))
                native_calls = ctypes.c_int()
            trials = []
            for trial in range(repeats + 1):
                calls[0] = 0
                start = time.perf_counter()
                if method == "arpack":
                    values, vectors = sla.eigsh(
                        op,
                        k=26,
                        ncv=64,
                        which="LA",
                        tol=args.arpack_tol,
                        maxiter=10000,
                        v0=initial.copy(),
                    )
                else:
                    count = lib.audit_spectra(
                        ctx,
                        args.arpack_tol,
                        initial.ctypes.data_as(dp),
                        native_values.ctypes.data_as(dp),
                        native_vectors.ctypes.data_as(dp),
                        ctypes.byref(native_calls),
                    )
                    if count != 26:
                        raise RuntimeError(f"Spectra converged {count}/26 modes")
                    vectors = native_vectors.T
                    calls[0] = native_calls.value
                elapsed = time.perf_counter() - start
                quality, _ = validate(original, vectors.T, reference)
                trials.append({"seconds": elapsed, "inverse_calls": calls[0], **quality})
                print(method, "trial", trial, elapsed, quality["accepted"], flush=True)
            report[method] = {
                "warmup": trials[0],
                "trials": trials[1:],
                "median_seconds": float(np.median([row["seconds"] for row in trials[1:]])),
                "all_accepted": all(row["accepted"] for row in trials),
                "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    Path("build/anvil_reference.npz").read_bytes()
                ).hexdigest(),
            }
        if args.backend in ("eigen", "pardiso"):
            lib.audit_free(ctx)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
