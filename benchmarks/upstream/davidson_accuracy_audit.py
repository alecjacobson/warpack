"""Untimed diagnosis of absolute residuals and roundoff in Davidson fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from warpack import KrylovSchur, SparseOperator

from .adapters import matrix
from .run_spectra import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("build/davidson-followup/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in range(4):
        case_id = f"DavidsonSymEigs_{case}"
        a = read(args.fixtures / f"{case_id}.a")
        if a.shape != (1000, 1000):
            parser.error(f"{case_id}: expected original 1000-by-1000 fixture")
        solver = KrylovSchur(
            SparseOperator(matrix(a)), 10, which="LA" if case % 2 == 0 else "SA", tol=1e-12
        )
        wp.capture_launch(solver.capture(300))
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        ax = solver.ax.numpy().T
        av = a @ v
        norms = np.linalg.norm(v, axis=0)
        # Diagnostic only: this host recomputation is not used by the solver.
        rq = np.sum(v * av, axis=0) / norms**2
        ref = la.eigvalsh(a)
        ref = ref[-10:][::-1] if case % 2 == 0 else ref[:10]
        row = dict(
            id=case_id,
            iterations=int(solver.iterations.numpy()[0]),
            recoveries=int(solver.recoveries.numpy()[0]),
            reported_residuals=solver.residuals.numpy().tolist(),
            physical_relative_residuals=(
                np.linalg.norm(av - v * w, axis=0) / np.maximum(1, abs(w))
            ).tolist(),
            norm_error=float(np.max(abs(norms - 1))),
            cached_product_error=float(np.max(abs(ax - av))),
            raw_residual=float(np.max(abs(av - v * w))),
            rayleigh_residual=float(np.max(abs((av - v * rq) / norms))),
            raw_value_error=float(np.max(abs(w - ref))),
            rayleigh_value_error=float(np.max(abs(rq - ref))),
        )
        rows.append(row)
        args.out.write_text(json.dumps(dict(cases=rows), indent=2) + "\n")
        print(case_id, row["raw_residual"], row["rayleigh_residual"], flush=True)


if __name__ == "__main__":
    main()
