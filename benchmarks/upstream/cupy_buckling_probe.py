"""Native-only CuPy diagnostics for reciprocal prestress buckling (#90).

No Warp, cuDSS, external streams, or graph capture. SuperLU is the optional
CuPy wrapper's own CPU factorization with GPU triangular applications; these
are diagnostic runs, not benchmark speed comparisons.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

import cupy as cp
import cupyx.scipy.linalg as dense
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import numpy as np
import scipy.optimize as opt

from benchmarks.applications.prestress_buckling import make_buckling_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--sign", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--methods", default="diagonal,none")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    methods = args.methods.split(",")
    if set(methods) - {"superlu", "dense", "diagonal", "none"}:
        parser.error("unknown method")
    case = make_buckling_case(args.side, args.modes, args.sign)
    p = case.problem
    assert not any(
        name == "warp" or name.startswith(("warp.", "warpack", "warp_cudss"))
        for name in sys.modules
    )
    a, b = cs.csr_matrix(args.sign * p.b), cs.csr_matrix(p.a)
    initial = cp.asarray(np.random.default_rng(2026).normal(size=(p.n, p.k)))
    report = dict(
        args=vars(args),
        n=p.n,
        k=p.k,
        versions={n: importlib.metadata.version(n) for n in ["numpy", "scipy", "cupy-cuda12x"]},
        native_only=True,
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                Path("benchmarks/applications/prestress_buckling.py"),
                Path(cl.lobpcg.__code__.co_filename),
            ]
        },
        rows=[],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        print("Starting", method, flush=True)
        if method == "superlu":
            factor = cl.splu(b.tocsc())
            pre = cl.LinearOperator(
                b.shape, matvec=factor.solve, matmat=factor.solve, dtype=cp.float64
            )
        elif method == "dense":
            factor = cp.linalg.cholesky(b.toarray())

            def apply(x):
                y = dense.solve_triangular(factor, x, lower=True)
                return dense.solve_triangular(factor, y, lower=True, trans="T")

            pre = cl.LinearOperator(b.shape, matvec=apply, matmat=apply, dtype=cp.float64)
        elif method == "diagonal":
            diagonal = 1 / b.diagonal()
            pre = cl.LinearOperator(
                b.shape,
                matvec=lambda x: diagonal * x,
                matmat=lambda x: diagonal[:, None] * x,
                dtype=cp.float64,
            )
        else:
            pre = None
        for trial in range(args.repeats):
            row = dict(method=method, trial=trial)
            try:
                cp.random.seed(2026)
                mu, v, history = cl.lobpcg(
                    a,
                    initial.copy(),
                    B=b,
                    M=pre,
                    largest=True,
                    tol=args.tol,
                    maxiter=args.steps,
                    retResidualNormsHistory=True,
                )
                w = args.sign / mu.get()
                v = v.get()
                kv, gv = p.a @ v, p.b @ v
                absolute = np.linalg.norm(gv - kv * (1 / w), axis=0)
                residual = np.linalg.norm(kv - gv * w, axis=0) / np.maximum(
                    1, np.linalg.norm(kv, axis=0) + abs(w) * np.linalg.norm(gv, axis=0)
                )
                orth = float(abs(v.T @ kv - np.eye(p.k)).max())
                remainder = v - p.requested_basis @ (p.requested_basis.T @ kv)
                space = float(np.sqrt(max(0, np.sum(remainder * (p.a @ remainder)))))
                cost = (
                    abs(w[:, None] - p.expected[None, :]) / np.maximum(1, abs(p.expected))[None, :]
                )
                ii, jj = opt.linear_sum_assignment(cost)
                spectrum = float(cost[ii, jj].max())
                row.update(
                    outcome="returned",
                    max_native_absolute_residual=float(absolute.max()),
                    history_length=len(history),
                    last_history_max=float(cp.max(cp.asarray(history[-1])).get()),
                    physical_residual=float(residual.max()),
                    orthogonality=orth,
                    requested_subspace_error=space,
                    spectrum_error=spectrum,
                    accepted=bool(
                        absolute.max() <= args.tol
                        and residual.max() < 1e-8
                        and orth < 1e-8
                        and space < 1e-8
                        and spectrum < 1e-8
                    ),
                )
            except Exception as error:
                row.update(
                    outcome="error",
                    reason=str(error),
                    traceback=traceback.format_exc(),
                    accepted=False,
                )
            report["rows"].append(row)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(
                method,
                trial,
                row["outcome"],
                row["accepted"],
                row.get("physical_residual", row.get("reason")),
                flush=True,
            )
    report["accepted"] = all(r["accepted"] for r in report["rows"])
    report["diagnostic_only"] = True
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
