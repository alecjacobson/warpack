"""Matched-start anvil timings: native CPU references, CuPy, and Warp inverses.

Run with OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4. Setup and warmup/capture
are separate; every reported solve excludes the final physical residual check.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp
import warp.optim.linear as linear

from benchmarks.audit_profile import graph_summary
from benchmarks.validate_dragon_meshes import mode_quality
from warpack import CGInverse, CuDSSInverse, KrylovSchur
from warpack.fem import assemble_rest, mass_normalized, read_mesh


def clocked(function, gpu=False):
    if gpu:
        wp.synchronize()
    start = time.perf_counter()
    value = function()
    if gpu:
        wp.synchronize()
    return time.perf_counter() - start, value


def validate(sa, vectors, reference):
    quality, vectors, values = mode_quality(sa, np.zeros(26), vectors)
    if reference is not None:
        rv, rw = reference
        cross = np.sum(rv[6:] * vectors[6:], axis=1)
        norms = np.sum(rv[6:] ** 2, axis=1) * np.sum(vectors[6:] ** 2, axis=1)
        mac = cross**2 / norms
        q = np.linalg.qr(rv[:6].T, mode="reduced")[0]
        r = np.linalg.qr(vectors[:6].T, mode="reduced")[0]
        quality.update(
            max_relative_elastic_eigenvalue_error=float(np.max(abs(values[6:] / rw[6:] - 1))),
            max_relative_elastic_frequency_error=float(
                np.max(abs(np.sqrt(values[6:] / rw[6:]) - 1))
            ),
            minimum_elastic_mac=float(mac.min()),
            rigid_subspace_min_cosine=float(np.linalg.svd(q.T @ r, compute_uv=False).min()),
        )
    quality["accepted"] = bool(
        quality["max_original_residual"] < 1e-7
        and quality["orthogonality_max_error"] < 1e-8
        and quality.get("minimum_elastic_mac", 1) > 0.99999
    )
    return quality, (vectors, values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["direct", "pcg"], default="direct")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pcg-tol", type=float, default=1e-12)
    p.add_argument("--pcg-maxiter", type=int, default=10000)
    p.add_argument("--out", default="results/anvil_benchmarks.json")
    args = p.parse_args()
    out = Path(args.out)
    mesh = Path("anvil for export-ftetwild.mesh")
    start = time.perf_counter()
    x, t = read_mesh(mesh)
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    shifted = mass_normalized(h, mass, shift=1)
    a.nnz_sync()
    shifted.nnz_sync()
    wp.synchronize()
    assembly = time.perf_counter() - start
    n = a.shape[0]
    sa = sp.bsr_matrix(
        (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()), shape=(n, n)
    ).tocsr()
    sa.sort_indices()
    report = (
        json.loads(out.read_text())
        if args.part == "pcg"
        else {
            "warp_runtime_version": wp.__version__,
            "warp_main_linear_revision": "5a0c33d17d847d0be07ae1003263b572aae0729a",
            "warp_linear_module_sha256": hashlib.sha256(
                Path(inspect.getfile(linear.preconditioner)).read_bytes()
            ).hexdigest(),
            "warp_environment": "official Warp 1.18.0+cu12 runtime with the entire unmodified linear.py from the pinned upstream main revision; CUDA-13 nightly cannot run on this driver",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["cupy-cuda12x", "numpy", "scipy", "nvidia-cudss-cu12"]
            },
            "mesh": str(mesh),
            "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
            "physical_problem": {
                "vertices": len(x),
                "tetrahedra": len(t),
                "normalized_longest_extent_m": 1.0,
                "young_modulus_pa": 1e5,
                "poisson_ratio": 0.3,
                "density_kg_m3": 1000.0,
                "mass": "lumped",
                "boundary_conditions": "free body",
                "hessian": "stable Neo-Hookean at rest",
            },
            "n": n,
            "k": 26,
            "elastic_modes": 20,
            "ncv": 64,
            "positive_inverse_shift": 1.0,
            "outer_tolerance": 1e-12,
            "repeats": args.repeats,
            "gpu": wp.get_device().name,
            "cpu": "Intel Xeon Platinum 8362",
            "cpu_blas_threads": 4,
            "assembly_io_jit_seconds": assembly,
            "protocol": "same start restored before every trial; one warmup then median of three solves; factor/preconditioner setup, JIT/warmup/capture, final original-problem validation, and transfers are outside solve timing; Warp graph initializes its basis on every replay; CuPy shares cuDSS factors; CPU references use separate native factors",
            "spectra_revision": "db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95",
            "methods": [],
        }
    )

    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")

    def emit(row):
        previous = [r for r in report["methods"] if r["method"] == row["method"]]
        if previous:
            report.setdefault("previous_trials", []).extend(previous)
            report["methods"] = [r for r in report["methods"] if r["method"] != row["method"]]
        report["methods"].append(row)
        save()
        print(json.dumps(row, indent=2), flush=True)

    reference = None
    if args.part == "direct":
        setup, inv = clocked(lambda: CuDSSInverse(shifted, 1), gpu=True)
        solver = KrylovSchur(inv, 26, ncv=64, which="LA", tol=1e-12)
        solver.initialize()
        initial = solver.views[0].numpy().ravel()
        np.save("build/anvil_initial.npy", initial)
        capture, graph = clocked(lambda: solver.capture(1, adaptive=False), gpu=True)
        samples = []
        for trial in range(args.repeats + 1):
            elapsed, _ = clocked(lambda: wp.capture_launch(graph), gpu=True)
            if trial:
                samples.append(elapsed)
        quality, reference = validate(sa, solver.eigenvectors.numpy(), None)
        assert quality["accepted"]
        np.savez("build/anvil_reference.npz", vectors=reference[0], values=reference[1])
        validation_seconds, _ = clocked(lambda: validate(sa, solver.eigenvectors.numpy(), None))
        emit(
            dict(
                method="Warp + cuDSS",
                setup_seconds=setup,
                warmup_capture_seconds=capture,
                solve_seconds=float(np.median(samples)),
                solve_samples_seconds=samples,
                inverse_applications=64,
                validation_host_seconds=validation_seconds,
                **quality,
            )
        )
        del graph, solver
        stream = cp.cuda.Stream.from_external(wp.get_stream())
        with stream:
            calls = [0]

            def mv(v):
                calls[0] += 1
                inv.apply(wp.from_dlpack(v.reshape(1, n)), inv.x)
                return cp.asarray(inv.x).reshape(n).copy()

            op = csl.LinearOperator((n, n), matvec=mv, dtype=np.float64)
            samples = []
            for trial in range(args.repeats + 1):
                v0 = cp.asarray(initial).copy()
                stream.synchronize()
                calls[0] = 0
                start = time.perf_counter()
                values, vectors = csl.eigsh(
                    op, k=26, ncv=64, which="LA", tol=1e-12, maxiter=10000, v0=v0
                )
                stream.synchronize()
                elapsed = time.perf_counter() - start
                if trial:
                    samples.append(elapsed)
                else:
                    warmup = elapsed
            quality, _ = validate(sa, cp.asnumpy(vectors).T, reference)
            emit(
                dict(
                    method="CuPy eigsh + shared cuDSS",
                    setup_seconds=setup,
                    warmup_seconds=warmup,
                    solve_seconds=float(np.median(samples)),
                    solve_samples_seconds=samples,
                    inverse_applications=calls[0],
                    **quality,
                )
            )
        # Also measure CuPy's native smallest-algebraic path without any inverse.
        # A failed bounded run is reported as such, never as an accepted speedup.
        from cupyx.scipy.sparse import csr_matrix

        with stream:
            ca = csr_matrix(sa)
            _ = ca @ cp.ones(n, dtype=cp.float64)
            stream.synchronize()
            samples = []
            for trial in range(args.repeats + 1):
                v0 = cp.asarray(initial).copy()
                stream.synchronize()
                start = time.perf_counter()
                values, vectors = csl.eigsh(
                    ca, k=26, ncv=64, which="SA", tol=1e-12, maxiter=2000, v0=v0
                )
                stream.synchronize()
                elapsed = time.perf_counter() - start
                quality, _ = validate(sa, cp.asnumpy(vectors).T, reference)
                print(
                    "Native CuPy SA trial",
                    trial,
                    elapsed,
                    quality["max_original_residual"],
                    flush=True,
                )
                if not quality["accepted"]:
                    samples = [elapsed]
                    break
                if trial:
                    samples.append(elapsed)
            emit(
                dict(
                    method="CuPy eigsh SA (no inverse, 2000-iteration budget)",
                    setup_seconds=0.0,
                    solve_seconds=float(np.median(samples)),
                    solve_samples_seconds=samples,
                    repeats=len(samples),
                    max_iterations=2000,
                    **quality,
                )
            )
        inv.close()
        del inv
        cpu = (sa + sp.eye(n, format="csr")).tocsc()
        cpu.sort_indices()
        setup, lu = clocked(
            lambda: sla.splu(
                cpu,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        )
        print("SuperLU factor seconds", setup, flush=True)
        calls = [0]

        def mv(v):
            calls[0] += 1
            return lu.solve(v)

        op = sla.LinearOperator((n, n), matvec=mv, dtype=np.float64)
        samples = []
        for trial in range(args.repeats + 1):
            calls[0] = 0
            v0 = initial.copy()
            elapsed, (values, vectors) = clocked(
                lambda: sla.eigsh(op, k=26, ncv=64, which="LA", tol=1e-12, maxiter=10000, v0=v0)
            )
            print("ARPACK trial", trial, elapsed, flush=True)
            if trial:
                samples.append(elapsed)
            else:
                warmup = elapsed
        quality, _ = validate(sa, vectors.T, reference)
        emit(
            dict(
                method="SciPy ARPACK + SuperLU",
                factor_options={
                    "permc_spec": "MMD_AT_PLUS_A",
                    "diag_pivot_thresh": 0.0,
                    "SymmetricMode": True,
                },
                setup_seconds=setup,
                warmup_seconds=warmup,
                solve_seconds=float(np.median(samples)),
                solve_samples_seconds=samples,
                inverse_applications=calls[0],
                **quality,
            )
        )
        del lu, op, vectors
        lib = ctypes.CDLL(str(Path("build/spectra_native.so").resolve()))
        ptr = ctypes.POINTER(ctypes.c_double)
        iptr = ctypes.POINTER(ctypes.c_int)
        lib.spectra_factor.argtypes = [ctypes.c_int, ctypes.c_int, iptr, iptr, ptr]
        lib.spectra_factor.restype = ctypes.c_void_p
        lib.spectra_solve.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ptr,
            ptr,
            ptr,
            iptr,
        ]
        lib.spectra_solve.restype = ctypes.c_int
        lib.spectra_free.argtypes = [ctypes.c_void_p]
        indptr = cpu.indptr.astype(np.int32)
        indices = cpu.indices.astype(np.int32)
        setup, ctx = clocked(
            lambda: lib.spectra_factor(
                n,
                cpu.nnz,
                indptr.ctypes.data_as(iptr),
                indices.ctypes.data_as(iptr),
                cpu.data.ctypes.data_as(ptr),
            )
        )
        assert ctx
        print("Eigen LDLT factor seconds", setup, flush=True)
        values = np.empty(26)
        vectors = np.empty((26, n))
        calls = ctypes.c_int()
        samples = []
        for trial in range(args.repeats + 1):
            elapsed, count = clocked(
                lambda: lib.spectra_solve(
                    ctx,
                    26,
                    64,
                    1e-12,
                    initial.ctypes.data_as(ptr),
                    values.ctypes.data_as(ptr),
                    vectors.ctypes.data_as(ptr),
                    ctypes.byref(calls),
                )
            )
            assert count == 26, count
            print("Spectra trial", trial, elapsed, flush=True)
            if trial:
                samples.append(elapsed)
            else:
                warmup = elapsed
        quality, _ = validate(sa, vectors, reference)
        emit(
            dict(
                method="Spectra + native Eigen LDLT",
                setup_seconds=setup,
                warmup_seconds=warmup,
                solve_seconds=float(np.median(samples)),
                solve_samples_seconds=samples,
                inverse_applications=calls.value,
                **quality,
            )
        )
        lib.spectra_free(ctx)
    else:
        data = np.load("build/anvil_reference.npz")
        reference = (data["vectors"], data["values"])
        setup, inv = clocked(
            lambda: CGInverse(
                shifted,
                tol=args.pcg_tol,
                maxiter=args.pcg_maxiter,
                preconditioner="block_jacobi_sequential",
            ),
            gpu=True,
        )
        solver = KrylovSchur(inv, 26, ncv=64, which="LA", tol=1e-12)
        print("Capturing block-Jacobi PCG eigensolve, including a full warmup", flush=True)
        capture, graph = clocked(lambda: solver.capture(1, adaptive=False), gpu=True)
        print("PCG capture seconds", capture, flush=True)
        wp.capture_debug_dot_print(graph, "build/anvil_pcg_graph.dot", verbose=True)
        samples = []
        stats = []
        enqueue_samples = []
        device_samples = []
        sample_windows = []
        sample_quality = []
        begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for trial in range(args.repeats + 1):
            inv.reset_stats()
            wp.synchronize()
            wp.record_event(begin)
            window_start = time.time()
            start = time.perf_counter()
            wp.capture_launch(graph)
            enqueue = time.perf_counter() - start
            wp.record_event(end)
            wp.synchronize()
            elapsed = time.perf_counter() - start
            window_end = time.time()
            device_elapsed = wp.get_event_elapsed_time(begin, end, False) / 1000
            counts = inv.stats.numpy().tolist()
            print("PCG trial", trial, elapsed, counts, flush=True)
            if trial:
                samples.append(elapsed)
                stats.append(counts)
                enqueue_samples.append(enqueue)
                device_samples.append(device_elapsed)
                sample_windows.append([window_start, window_end])
                trial_quality, _ = validate(sa, solver.eigenvectors.numpy(), reference)
                sample_quality.append(trial_quality)
                print(
                    "PCG original-problem residual",
                    trial_quality["max_original_residual"],
                    flush=True,
                )
            else:
                warmup = elapsed
                preliminary, _ = validate(sa, solver.eigenvectors.numpy(), reference)
                print(
                    "PCG warmup original-problem residual",
                    preliminary["max_original_residual"],
                    flush=True,
                )
        quality, _ = validate(sa, solver.eigenvectors.numpy(), reference)
        quality["accepted"] = all(q["accepted"] for q in sample_quality)
        for key in ["max_original_residual", "max_elastic_residual", "orthogonality_max_error"]:
            quality[key] = max(q[key] for q in sample_quality)
        emit(
            dict(
                method="Warp + 3x3 block-Jacobi PCG",
                preconditioner="Warp upstream block_jacobi_sequential",
                setup_seconds=setup,
                warmup_capture_seconds=capture,
                warmup_seconds=warmup,
                solve_seconds=float(np.median(samples)),
                solve_samples_seconds=samples,
                solve_sample_quality=sample_quality,
                measurement_unix_windows=sample_windows,
                process_id=os.getpid(),
                inner_relative_tolerance=args.pcg_tol,
                inner_max_iterations=args.pcg_maxiter,
                inner_statistics_fields=[
                    "total_iterations",
                    "maximum_iterations_per_rhs",
                    "unconverged_rhs",
                    "inverse_applications",
                ],
                inner_statistics=stats,
                host_enqueue_seconds=float(np.median(enqueue_samples)),
                device_solve_seconds=float(np.median(device_samples)),
                captured_graph=graph_summary("build/anvil_pcg_graph.dot"),
                **quality,
            )
        )


if __name__ == "__main__":
    main()
