"""Matched-accuracy regression for Spectra issue #7."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from benchmarks.applications.compare import other_gpu_processes
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, GeneralizedEigensolver, GeneralizedKrylovSchur, SparseOperator


def main():
    Path("results").mkdir(exist_ok=True)
    p = np.load("tests/fixtures/spectra_generalized_shift_5.npz")
    a, b = p["a"], p["b"]
    sigma = 1.2345
    k = 10
    ref = la.eigvalsh(a, b)
    ref = ref[np.argsort(abs(ref - sigma))[:k]]
    rows = []
    for cls, width, steps in [
        (GeneralizedEigensolver, 20, 100),
        (GeneralizedEigensolver, 20, 300),
        (GeneralizedKrylovSchur, 1, 20),
    ]:
        inv = CuDSSInverse(matrix(a - sigma * b), width, mtype="symmetric")
        s = cls(
            SparseOperator(matrix(a)),
            SparseOperator(matrix(b)),
            inv,
            k,
            ncv=20,
            target=sigma,
            tol=1e-12,
        )
        g = s.capture(steps)
        wp.capture_launch(g)
        wp.synchronize()
        ts = []
        checks = []
        others = set()
        for _ in range(3):
            others.update(other_gpu_processes())
            wp.synchronize()
            t = time.perf_counter()
            wp.capture_launch(g)
            wp.synchronize()
            ts.append(time.perf_counter() - t)
            others.update(other_gpu_processes())
            w = s.eigenvalues.numpy()
            v = s.eigenvectors.numpy().T
            checks.append(
                dict(
                    residual=float(abs(a @ v - b @ v * w).max()),
                    spectrum_error=float(abs(np.sort(w) - np.sort(ref)).max()),
                    mass_orthogonality=float(abs(v.T @ b @ v - np.eye(k)).max()),
                    converged=int(s.converged.numpy()[0]),
                )
            )
        rows.append(
            dict(
                solver=cls.__name__,
                cycles=steps,
                samples=ts,
                median=float(np.median(ts)),
                checks=checks,
                other_gpu_processes=sorted(others),
                timing_uncontended=not others,
            )
        )
        s.graph = None
        del g, s
        gc.collect()
        inv.close()
    Path("results/generalized_krylov_regression.json").write_text(
        json.dumps(
            dict(
                case="SymGEigsShift_5",
                n=100,
                k=k,
                ncv=20,
                target=sigma,
                gpu=wp.get_device().name,
                warp=wp.__version__,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
