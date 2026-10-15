"""Native CuPy-only reproduction of the rank-deficient string SVD comparison.

No Warp import, graph, shared stream, or interop in this probe. Host construction
and checks are independent of the benchmark adapter; timings are not measured.
"""

import argparse
import json
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/pod_cupy_probe.json")
    args = parser.parse_args()
    rows = []
    for size in [10000, 100000]:
        frequency = np.arange(1, 23)
        u = np.sin(np.pi * (np.arange(1, size + 1) / (size + 1))[:, None] * frequency) * np.sqrt(
            2 / (size + 1)
        )
        v = np.cos(np.pi * (np.arange(48) / 24)[:, None] * frequency) * np.sqrt(2 / 48)
        spectrum = np.geomspace(20, 0.05, 22)
        host = (u * spectrum) @ v.T
        a = cs.csr_matrix(cp.asarray(host))
        for ncv in [24, 32, 40, 47]:
            cp.random.seed(2026)
            left, values, right = cl.svds(a, k=20, ncv=ncv, tol=1e-11, maxiter=10000)
            w, x, y = values.get(), left.get(), right.get().T
            error = float(np.max(abs(np.sort(w)[::-1] - spectrum[:20])))
            residual = float(
                max(
                    np.linalg.norm(host @ y - x * w, axis=0).max(),
                    np.linalg.norm(host.T @ x - y * w, axis=0).max(),
                )
                / 20
            )
            # Reproduce the normal-equation stage with the same random stream.
            operator = cl.aslinearoperator(a)
            cp.random.seed(2026)
            squared, basis = cl.eigsh(
                operator.H @ operator, k=20, ncv=ncv, tol=1e-11, maxiter=10000
            )
            raw = squared.get()
            row = dict(
                size=size,
                k=20,
                ncv=ncv,
                values=w.tolist(),
                value_error=error,
                residual=residual,
                accepted=bool(error < 1e-8 and residual < 1e-8),
                raw_normal_eigenvalues=raw.tolist(),
                raw_normal_finite=bool(np.isfinite(raw).all()),
            )
            rows.append(row)
            print(size, ncv, row["accepted"], error, residual, row["raw_normal_finite"], flush=True)
    # JSON null for non-finite raw diagnostic eigenvalues, preserving the flag.
    for row in rows:
        row["raw_normal_eigenvalues"] = [
            x if np.isfinite(x) else None for x in row["raw_normal_eigenvalues"]
        ]
    Path(args.out).write_text(
        json.dumps(
            dict(
                cupy=cp.__version__,
                numpy=np.__version__,
                protocol="Independent native CuPy default-stream reproduction; no Warp imported; no timing claims; k20, seed2026, maxiter10000, tolerance1e-11; raw eigsh stage inspected separately after resetting seed.",
                cases=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
