"""Warp/CuPy on every original Spectra benchmark dataset (10 warm trials)."""

import argparse
import hashlib
import importlib.util
import json
import time

import cupy as cp
import numpy as np
import warp as wp
from cupyx.scipy import sparse as cs

from benchmarks.runtime import warp_runtime_hashes
from warpack import GeneralEigensolver, KrylovSchur, SparseOperator
from warpack.arnoldi import scale_vector

from .adapters import complex_matrix, execute, matrix, output
from .general_reference import GeneralReference
from .run_spectra import ROOT, read


class StartedGeneral(GeneralEigensolver):
    """Benchmark-only start adapter; identical Warp initialization after copy."""

    def initialize(self, seed=42):
        if self._checkpoint is not None:
            self._checkpoint.reset()
        self.restart_index.fill_(-1)
        self.residual_restarts.zero_()
        self.residual_fallbacks.zero_()
        self.status.zero_()
        wp.copy(self.qrows[0], self.start_vector)
        self.length(0, self.norm)
        wp.launch(scale_vector, self.n, [self.qrows[0], self.norm], device=self.device)
        for j in range(1, self.keep):
            self.expand(j)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--out", default="results/upstream_benchmarks.json")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "pinned_cupy_eigen", ROOT / "build/cupy_upstream_eigen.py"
    )
    eigen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eigen)
    folder = ROOT / "build/spectra-benchmark-fixtures"
    cases = [
        json.loads(line)
        for p in sorted(folder.glob("*.jsonl"))
        for line in p.read_text().splitlines()
    ]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    report = dict(
        repeats=10,
        spectra_revision="db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95",
        cupy_revision="2316a02d70c7e39d6a7c035529df8358d794256e",
        cupy_runtime=cp.__version__,
        warp=wp.__version__,
        gpu=wp.get_device().name,
        runtime_hashes=warp_runtime_hashes("build/application_spectra.so"),
        source_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "warpack/arnoldi.py",
                "warpack/schur.py",
                "warpack/schur_tiled.py",
                "benchmarks/upstream/run_benchmarks.py",
            )
        },
        protocol="Original Spectra generators and starts, all 40 datasets. GPU setup/JIT/capture and validation excluded from 10 warm synchronized solves. CuPy uses pinned _eigen.py over installed runtime. General Warp initialization adapted only to accept original v0. All backends request tol=1e-10; convergence criteria differ by implementation. General Warp trials additionally check every timed replay against independent requested spectra/subspaces and vector norms, allowing either conjugate member at the cutoff while requiring all larger modes.",
        cases=[],
    )
    for c in cases:
        a = read(folder / (c["id"] + ".a"))
        v0 = read(folder / (c["id"] + ".v0")).ravel()
        n = len(a)
        k = c["k"]
        m = c["ncv"]
        reference = GeneralReference(a, k) if c["family"] == "benchmark_gen" else None
        results = []
        for backend in ("warp", "cupy"):
            if backend == "cupy" and c["family"] == "benchmark_gen":
                results.append(
                    dict(
                        backend=backend,
                        outcome="unsupported",
                        reason="CuPy sparse linalg has no general eigs",
                    )
                )
                continue
            try:
                start = time.perf_counter()
                if backend == "warp":
                    if c["family"] == "benchmark_sym":
                        solver = KrylovSchur(
                            SparseOperator(matrix(a)),
                            k,
                            ncv=m,
                            which="LM",
                            tol=1e-10,
                            initial_vector=wp.array(v0, dtype=wp.float64),
                        )
                    else:
                        solver = StartedGeneral(complex_matrix(a), k, ncv=m, which="LM", tol=1e-10)
                        solver.start_vector = wp.array(
                            np.column_stack([v0, np.zeros(n)]), dtype=wp.vec2d
                        )
                    wp.synchronize()
                    setup = time.perf_counter() - start

                    def validate():
                        trial = reference.check(
                            output(solver.eigenvalues), output(solver.eigenvectors).T
                        )
                        trial["converged"] = int(solver.converged.numpy()[0])
                        trial["status"] = int(solver.status.numpy()[0])
                        trial["iterations"] = int(solver.iterations.numpy()[0])
                        trial["accepted"] &= trial["converged"] == k and trial["status"] == 0
                        return trial

                    timings = execute(
                        solver,
                        300,
                        repeats=10,
                        validate=validate if reference is not None else None,
                    )
                    w = output(solver.eigenvalues)
                    v = output(solver.eigenvectors).T
                    converged = int(solver.converged.numpy()[0])
                    status = int(solver.status.numpy()[0])
                else:
                    ca = cs.csr_matrix(cp.asarray(a))
                    cv = cp.asarray(v0)
                    cp.cuda.get_current_stream().synchronize()
                    setup = time.perf_counter() - start
                    eigen.eigsh(ca, k=k, ncv=m, v0=cv.copy(), which="LM", tol=1e-10)
                    cp.cuda.get_current_stream().synchronize()
                    samples = []
                    for _ in range(10):
                        start_vector = cv.copy()
                        cp.cuda.get_current_stream().synchronize()
                        start = time.perf_counter()
                        w, v = eigen.eigsh(ca, k=k, ncv=m, v0=start_vector, which="LM", tol=1e-10)
                        cp.cuda.get_current_stream().synchronize()
                        samples.append(time.perf_counter() - start)
                    timings = dict(solve_seconds=float(np.median(samples)), samples=samples)
                    w = w.get()
                    v = v.get()
                    converged = len(w)
                    status = 0
                residual = float(np.max(np.abs(a @ v - v * w)))
                results.append(
                    dict(
                        backend=backend,
                        outcome="passed"
                        if np.isfinite(residual)
                        and residual < 1e-8
                        and converged == k
                        and status == 0
                        and all(t["accepted"] for t in timings.get("trials", []))
                        else "failed",
                        setup_seconds=setup,
                        **timings,
                        max_absolute_residual=residual,
                        converged=converged,
                        status=status,
                    )
                )
            except Exception as exc:
                import traceback

                results.append(
                    dict(
                        backend=backend,
                        outcome="error",
                        reason=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )
        report["cases"].append(
            dict(
                **c,
                n=n,
                results=results,
                matrix_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                start_sha256=hashlib.sha256(v0.tobytes()).hexdigest(),
                requested_spectrum_checked=reference is not None,
                all_timed_replays_validated=reference is not None,
            )
        )
        (ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(
            c["id"],
            n,
            [(r["backend"], r["outcome"], r.get("solve_seconds")) for r in results],
            flush=True,
        )


if __name__ == "__main__":
    main()
