"""Diagnostic inverse-space precision probe; original physical gates stay mandatory.

Requires optional mpmath for an 80-digit inverse of the exact stored FP64 matrix.
This is validation/reference computation only, never a production solve path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import warp as wp

from benchmarks.acceptance import errors_below_tolerance
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--refinements", default="0,1,2,3")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    refinements = [int(x) for x in args.refinements.split(",")]
    if args.instances < 1 or args.replays < 1 or not refinements or min(refinements) < 0:
        parser.error("positive instance/replay counts and nonnegative correction budgets required")
    mp.mp.dps = 80
    n, sigma = 20, -1e-6
    eye = np.eye(n)
    a = eye - 0.5 * np.roll(eye, 1, axis=0) - 0.5 * np.roll(eye, -1, axis=0)
    shifted = a - sigma * eye
    exact = mp.matrix(shifted.tolist()) ** -1
    expected, basis = np.linalg.eigh(a)
    report = dict(
        args=vars(args),
        diagnostic=True,
        protocol="Two fresh factor instances and two replays by default; identical Warp seed42 starts. 100 fixed cycles, tol1e-12 for inverse estimates/checks; original physical1e-9. All attempts retained; no timings. Exact inverse uses80-digit arithmetic on the stored FP64 entries. It only validates returned vectors; never initializes the solve.",
        n=n,
        sigma=sigma,
        matrix_sha256=hashlib.sha256(shifted.tobytes()).hexdigest(),
        condition=float(np.linalg.cond(shifted)),
        precision_digits=80,
        gpu=wp.get_device().name,
        warp=wp.__version__,
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path("warpack/solver.py"), Path("warpack/kernels.py")]
        },
        rows=[],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for corrections in refinements:
        for k, ncv in [(3, 6), (5, 12), (6, 12)]:
            for factor_instance in range(args.instances):
                inverse = CuDSSInverse(
                    matrix(shifted), 1, mtype="symmetric", refinement_steps=corrections
                )
                solver = KrylovSchur(inverse, k, ncv=ncv, tol=1e-12)
                physical = EigenpairEvaluation(
                    SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-9
                )
                fresh = EigenpairEvaluation(
                    inverse,
                    solver.eigenvectors,
                    eigenvalues=solver.eigenvalues,
                    rhs_width=1,
                    tol=solver.tol,
                )

                def finalize():
                    physical.run()
                    fresh.run()

                graph = solver.capture(100, finalizer=finalize)
                try:
                    for replay in range(args.replays):
                        wp.capture_launch(graph)
                        v = solver.eigenvectors.numpy()
                        mu = solver.eigenvalues.numpy()
                        w = physical.values.numpy()
                        transformed_values = sigma + 1 / mu
                        transformed_residuals = np.linalg.norm(
                            a @ v.T - v.T * transformed_values, axis=0
                        ) / np.maximum(1, abs(transformed_values))
                        fv = fresh.ax.numpy()
                        ideal_res, application_error = [], []
                        for vector, value, fx in zip(v, mu, fv):
                            mx = mp.matrix(vector.tolist())
                            ex = exact * mx
                            ideal_res.append(
                                float(mp.norm(ex - mp.mpf(float(value)) * mx) / max(1, abs(value)))
                            )
                            application_error.append(
                                float(mp.norm(mp.matrix(fx.tolist()) - ex) / max(1, abs(value)))
                            )
                        allowed = basis[:, expected <= expected[k - 1] + 1e-9]
                        row = dict(
                            refinement_steps=corrections,
                            k=k,
                            ncv=ncv,
                            factor_instance=factor_instance,
                            replay=replay,
                            inverse_count=int(solver.converged.numpy()[0]),
                            fresh_count=int(fresh.converged.numpy()[0]),
                            status=int(solver.status.numpy()[0]),
                            physical_count=int(physical.converged.numpy()[0]),
                            cached_inverse_residuals=solver.residuals.numpy().tolist(),
                            fresh_inverse_residuals=fresh.residuals.numpy().tolist(),
                            ideal_inverse_residuals=ideal_res,
                            inverse_application_error=application_error,
                            physical_residuals=physical.residuals.numpy().tolist(),
                            returned_physical_residuals=transformed_residuals.tolist(),
                            returned_spectrum_error=float(
                                abs(np.sort(transformed_values) - expected[:k]).max()
                            ),
                            inverse_values=mu.tolist(),
                            rayleigh_values=w.tolist(),
                            vectors=v.tolist(),
                            spectrum_error=float(abs(np.sort(w) - expected[:k]).max()),
                            subspace_error=float(np.linalg.norm(v.T - allowed @ (allowed.T @ v.T))),
                            orthogonality_error=float(abs(v @ v.T - np.eye(k)).max()),
                        )
                        row["physical_accepted"] = bool(
                            row["status"] == 0
                            and row["physical_count"] == k
                            and errors_below_tolerance(row["physical_residuals"], 1e-9)
                            and errors_below_tolerance(row["returned_physical_residuals"], 1e-9)
                            and row["returned_spectrum_error"] < 1e-9
                            and row["spectrum_error"] < 1e-9
                            and row["subspace_error"] < 1e-9
                            and row["orthogonality_error"] < 1e-10
                        )
                        row["cached_full_but_fresh_failed"] = (
                            row["inverse_count"] == k and row["fresh_count"] < k
                        )
                        report["rows"].append(row)
                        out.write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            corrections,
                            k,
                            factor_instance,
                            replay,
                            row["inverse_count"],
                            row["fresh_count"],
                            row["physical_accepted"],
                            flush=True,
                        )
                finally:
                    solver.graph = graph = None
                    wp.synchronize_device()
                    inverse.close()
    report["physical_accepted"] = all(row["physical_accepted"] for row in report["rows"])
    report["inverse_all_pass"] = all(row["fresh_count"] == row["k"] for row in report["rows"])
    out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["physical_accepted"]:
        raise SystemExit("Original physical contract failed; all diagnostic rows retained")


if __name__ == "__main__":
    main()
