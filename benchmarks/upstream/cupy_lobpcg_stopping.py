"""Native CuPy stopping diagnostic: no Warp solver or preconditioner involved."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cs
import numpy as np
from cupyx.scipy.sparse.linalg import lobpcg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["diagonal", "cycles"], default="diagonal")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.tol <= 0 or not np.isfinite(args.tol) or args.repeats < 1 or args.maxiter < 1:
        p.error("require finite positive tol and positive repeats/maxiter")
    n, k = 128, 6
    if args.case == "diagonal":
        diagonal = np.repeat(np.arange(1.0, 33.0), 4)
        a = cs.diags(cp.asarray(diagonal), format="csr")

        def reference_product(v):
            return diagonal[:, None] * v

        expected = diagonal[:k]
    else:
        cycle = np.diag(np.full(32, 2.0))
        cycle += np.diag(np.full(31, -1.0), 1) + np.diag(np.full(31, -1.0), -1)
        cycle[0, -1] = cycle[-1, 0] = -1
        host = np.kron(np.eye(4), cycle)
        a = cs.csr_matrix(cp.asarray(host))

        def reference_product(v):
            return host @ v

        expected = np.sort(np.tile(2 - 2 * np.cos(2 * np.pi * np.arange(32) / 32), 4))[:k]
    start = np.random.default_rng(args.seed).normal(size=(n, k))
    rows = []
    for _ in range(args.repeats):
        w, v, history = lobpcg(
            a,
            cp.asarray(start.copy()),
            largest=False,
            tol=args.tol,
            maxiter=args.maxiter,
            retResidualNormsHistory=True,
        )
        values, vectors = w.get(), v.get()
        history = cp.stack(history).get()
        residuals = np.linalg.norm(reference_product(vectors) - vectors * values, axis=0)
        rows.append(
            dict(
                history_length=len(history),
                last_history_absolute_residuals=history[-1].tolist(),
                minimum_history_by_column=history.min(axis=0).tolist(),
                returned_absolute_residuals=residuals.tolist(),
                native_tolerance_met=bool(np.all(residuals <= args.tol)),
                all_columns_previously_below_tolerance=bool(
                    np.all(history.min(axis=0) <= args.tol)
                ),
                orthogonality=float(abs(vectors.T @ vectors - np.eye(k)).max()),
                eigenvalue_error=float(abs(np.sort(values) - expected).max()),
            )
        )
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = dict(
        args=vars(args),
        n=n,
        k=k,
        gpu=device["name"].decode(),
        driver_version=cp.cuda.runtime.driverGetVersion(),
        cuda_runtime_version=cp.cuda.runtime.runtimeGetVersion(),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "cupy-cuda12x")},
        rows=rows,
        scope="Native CuPy only, no preconditioner, no Warp. Accuracy diagnostic, not timings.",
        note="History columns can rotate within repeated eigenspaces; minima are not physical mode identities. Fresh returned-pair residuals determine native-tolerance status.",
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
