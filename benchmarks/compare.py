"""Matched-matrix CPU/GPU benchmark; NumPy/SciPy/CuPy are reference code only."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp
import warp.sparse as ws

from warpack import CuDSSInverse, KrylovSchur, SparseOperator, SymmetricEigensolver


def wp_matrix(a):
    coo = a.tocoo()
    return ws.bsr_from_triplets(
        *a.shape,
        wp.array(coo.row, dtype=wp.int32),
        wp.array(coo.col, dtype=wp.int32),
        wp.array(coo.data, dtype=wp.float64),
    )


def quality(a, w, v):
    r = np.linalg.norm(a @ v - v * w, axis=0)
    return dict(
        residual=float(r.max()),
        relative_residual=float((r / np.maximum(1, np.abs(w))).max()),
        orthogonality=float(np.max(np.abs(v.T @ v - np.eye(len(w))))),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/benchmarks.json")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    rng = np.random.default_rng(123)
    dense = rng.normal(size=(1000, 1000))
    dense += dense.T
    r = sp.random(
        10000,
        10000,
        density=0.001,
        random_state=rng,
        data_rvs=lambda n: rng.uniform(-0.5, 0.5, n),
    )
    r = (r + r.T).tocsr()
    line = sp.diags([-np.ones(63), np.full(64, 2.0), -np.ones(63)], [-1, 0, 1])
    poisson = sp.kronsum(line, line).tocsr()
    cases = [
        ("dense_sym_1000", sp.csr_matrix(dense), "LM", 50),
        ("sparse_sym_10000", r, "LM", 50),
        ("poisson_4096", poisson, "SA", 30),
    ]
    line3 = sp.diags([-np.ones(49), np.full(50, 2.0), -np.ones(49)], [-1, 0, 1])
    large = sp.kronsum(sp.kronsum(line3, line3), line3).tocsr()
    large = large + sp.diags(rng.uniform(0.0, 0.2, large.shape[0]))
    cases.append(("heterogeneous_laplacian_125000", large, "LA", 100))
    report = dict(
        hardware=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        cupy=cp.__version__,
        warp=wp.__version__,
        repeats=args.repeat,
        cases=[],
    )
    for name, a, which, iters in cases:
        k = 20
        m = 48
        print(name, flush=True)
        rows = []
        d = csp.csr_matrix(a)
        wa = wp_matrix(a)
        setup_start = time.perf_counter()
        inv = CuDSSInverse(wa, m) if which == "SA" else None
        wp.synchronize()
        setup = time.perf_counter() - setup_start
        s = (
            SymmetricEigensolver(
                SparseOperator(wa),
                k,
                ncv=m,
                which=which,
                iteration_operator=inv,
                tol=1.0e-11,
            )
            if inv
            else KrylovSchur(SparseOperator(wa), k, ncv=m, which=which, tol=1.0e-11)
        )
        s.solve(1)
        wp.synchronize()
        capture_start = time.perf_counter()
        graph = s.capture(iters if inv else 100, adaptive=not bool(inv))
        capture_seconds = time.perf_counter() - capture_start
        timings = []
        for i in range(args.repeat + 1):
            t = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            elapsed = time.perf_counter() - t
            if i:
                timings.append(elapsed)
        row = dict(
            method="warpack" + ("+cuDSS" if inv else ""),
            seconds=float(np.median(timings)),
            setup_seconds=setup,
            capture_seconds=capture_seconds,
            iterations=int(s.iterations.numpy()[0]),
            converged=int(s.converged.numpy()[0]),
            graph=True,
            eigenvalues=s.eigenvalues.numpy().tolist(),
            **quality(a, s.eigenvalues.numpy(), s.eigenvectors.numpy().T),
        )
        rows.append(row)
        print(row, flush=True)
        for method in ["scipy.eigsh", "cupy.eigsh", "cupy.lobpcg"]:
            timings = []
            try:
                for repeat in range(args.repeat + 1):
                    cp.cuda.get_current_stream().synchronize()
                    t = time.perf_counter()
                    if method == "scipy.eigsh":
                        w, v = sla.eigsh(a, k=k, which=which, tol=1.0e-10, ncv=m, maxiter=10000)
                    elif method == "cupy.eigsh":
                        w, v = csl.eigsh(d, k=k, which=which, tol=1.0e-9, ncv=m, maxiter=10000)
                    else:
                        if which == "LM":
                            break
                        w, v = csl.lobpcg(
                            d,
                            cp.asarray(rng.normal(size=(len(d.indptr) - 1, k))),
                            largest=(which == "LA"),
                            tol=1.0e-9,
                            maxiter=1000,
                        )
                    cp.cuda.get_current_stream().synchronize()
                    elapsed = time.perf_counter() - t
                    if repeat:
                        timings.append(elapsed)
                if not timings:
                    continue
                if method.startswith("cupy"):
                    w, v = cp.asnumpy(w), cp.asnumpy(v)
                row = dict(
                    method=method,
                    seconds=float(np.median(timings)),
                    eigenvalues=w.tolist(),
                    **quality(a, w, v),
                )
                rows.append(row)
                print(row, flush=True)
            except Exception as e:
                rows.append(dict(method=method, error=str(e)))
        path = Path("build") / (name + ".mtx")
        scipy.io.mmwrite(path, a, symmetry="general")
        try:
            runs = [
                json.loads(
                    subprocess.check_output(
                        ["build/spectra_bench", str(path), str(k), which], text=True
                    )
                )
                for _ in range(args.repeat)
            ]
            row = runs[0]
            row["seconds"] = float(np.median([r["seconds"] for r in runs]))
            rows.append(row)
            print(row, flush=True)
        except Exception as e:
            rows.append(dict(method="Spectra", error=str(e)))
        reference = next(
            (
                np.sort(r["eigenvalues"])
                for r in rows
                if r.get("method") == "scipy.eigsh" and "eigenvalues" in r
            ),
            None,
        )
        if reference is not None:
            for row in rows:
                if "eigenvalues" in row:
                    row["spectrum_error"] = float(
                        np.max(np.abs(np.sort(row["eigenvalues"]) - reference))
                    )
        report["cases"].append(
            dict(name=name, n=a.shape[0], nnz=a.nnz, k=k, which=which, results=rows)
        )
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        if inv:
            inv.close()


if __name__ == "__main__":
    main()
