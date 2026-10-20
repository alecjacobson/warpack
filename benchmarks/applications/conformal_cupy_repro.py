"""CuPy reference isolation without Warp/cuDSS.

Small controls use native CuPy dense Cholesky preconditioning. The optional
large control uses a host SuperLU inverse with explicit transfers. Neither is
a timing comparison or a proposed scalable production solver.
"""

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path

import cupy as cp
import cupyx.scipy.linalg as dense
import cupyx.scipy.sparse as sparse
import cupyx.scipy.sparse.linalg as linear
import numpy as np
import scipy.sparse.linalg as sla

from .conformal import boundary_reference, make_conformal_case, metrics


def run(side, scaled, repeats=2, host_factor=False):
    case = make_conformal_case(
        side, 20 if host_factor else 6, 0.25, shift=-0.001 if host_factor else None
    )
    reference = boundary_reference(case)
    k, m = case.reduced
    a = k - case.shift * m
    scale = -case.shift if scaled else 1.0
    tolerance = 1e-11 if scaled else 1e-9
    metric = sparse.csr_matrix(a)
    operator = sparse.csr_matrix(scale * m)
    if host_factor:
        factor = sla.splu(a.tocsc())

        def solve(x):
            return cp.asarray(factor.solve(x.get()))

        preconditioner = (
            "host SciPy SuperLU isolation; explicit GPU/CPU transfers; NOT a timing comparison"
        )
    else:
        factor = cp.linalg.cholesky(cp.asarray(a.toarray()))

        def solve(x):
            y = dense.solve_triangular(factor, x, lower=True)
            return dense.solve_triangular(factor, y, lower=True, trans="T")

        preconditioner = "native CuPy dense Cholesky and triangular solves"

    inverse = linear.LinearOperator(a.shape, matvec=solve, matmat=solve, dtype=cp.float64)
    start = cp.asarray(np.random.default_rng(2026).normal(size=(case.n, case.k)))
    record = dict(
        side=side,
        n=case.n,
        k=case.k,
        shift=case.shift,
        scale=scale,
        tolerance=tolerance,
        seed=2026,
        preconditioner=preconditioner,
        trials=[],
    )
    for _ in range(repeats):
        try:
            cp.random.seed(2026)
            mu, vectors, history = linear.lobpcg(
                operator,
                start.copy(),
                B=metric,
                M=inverse,
                largest=True,
                tol=tolerance,
                maxiter=300,
                retResidualNormsHistory=True,
            )
            fresh = cp.linalg.norm(operator @ vectors - (metric @ vectors) * mu, axis=0)
            physical = case.shift + scale / mu.get()
            full = case.lift((vectors / cp.sqrt(mu / scale)).get())
            check = metrics(case, physical, full, reference)
            check.update(
                native_history_max=float(history[-1].max().get()),
                native_fresh_max=float(fresh.max().get()),
                iterations=len(history),
            )
            check["native_stopping_passed"] = (
                max(check["native_history_max"], check["native_fresh_max"]) <= tolerance
            )
            record["trials"].append(check)
        except Exception as error:
            record["trials"].append(
                dict(accepted=False, error=repr(error), traceback=traceback.format_exc())
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--large-host",
        action="store_true",
        help="Isolate the large scaled failure with a host SuperLU inverse; no dense volume matrix",
    )
    args = parser.parse_args()
    report = dict(
        cupy=cp.__version__,
        scipy=importlib.metadata.version("scipy"),
        note="Diagnostic only; no Warp/cuDSS imported. Installed CuPy source unchanged.",
        cases=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configurations = (
        [(128, True)] if args.large_host else [(s, t) for s in (16, 24, 32) for t in (False, True)]
    )
    for side, scaled in configurations:
        row = run(side, scaled, host_factor=args.large_host)
        report["cases"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            side, scaled, [(r.get("accepted"), r.get("error")) for r in row["trials"]], flush=True
        )


if __name__ == "__main__":
    main()
