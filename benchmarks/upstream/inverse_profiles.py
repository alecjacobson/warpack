"""Compare cuDSS factor settings on the exact sparse Spectra SM fixtures."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.upstream.adapters import matrix
from benchmarks.upstream.run_spectra import FIX, read
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", action="store_true")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument(
        "--reordering", default="default", choices=["default", "colamd", "btf_colamd"]
    )
    parser.add_argument("--mtype", choices=["symmetric", "general"], default="symmetric")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out", default="results/inverse_profiles.json")
    args = parser.parse_args()

    cases = [json.loads(line) for p in FIX.glob("*.jsonl") for line in p.read_text().splitlines()]
    report = dict(
        protocol="Exact Spectra real SM fixtures; original 100-cycle budget and absolute 1e-9 residual gate; explicit fixed correction budgets",
        cases=[],
    )
    for key in ["SymEigs_17", "SymEigs_22", "SymEigs_27"]:
        c = next(c for c in cases if c["id"] == key)
        a = read(FIX / (key + ".a"))
        ref = np.linalg.eigvalsh(a)
        wanted = ref[np.argsort(abs(ref))[: c["k"]]]
        b = np.random.default_rng(42).normal(size=(1, len(a)))
        for steps in args.steps:
            inv = CuDSSInverse(
                matrix(a),
                1,
                mtype=args.mtype,
                refinement_steps=steps,
                reordering=args.reordering,
                matching=args.matching,
            )
            rhs = wp.array(b, dtype=wp.float64)
            out = wp.empty_like(rhs)
            inv.apply(rhs, out)
            linear_error = float(np.linalg.norm(out.numpy() @ a - b) / np.linalg.norm(b))
            if args.linear_only:
                row = dict(
                    id=key,
                    mtype=args.mtype,
                    reordering=args.reordering,
                    matching=args.matching,
                    refinement_steps=steps,
                    linear_relative_residual=linear_error,
                )
                report["cases"].append(row)
                Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
                print(row, flush=True)
                inv.close()
                continue
            solver = KrylovSchur(inv, c["k"], ncv=c["ncv"], tol=1e-12)
            ev = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-12)
            graph = solver.capture(100, finalizer=ev.run)
            wp.capture_launch(graph)
            values, vectors = ev.values.numpy(), solver.eigenvectors.numpy().T
            residual = float(abs(a @ vectors - vectors * values).max())
            spectrum = float(abs(np.sort(values) - np.sort(wanted)).max())
            row = dict(
                id=key,
                mtype=args.mtype,
                reordering=args.reordering,
                matching=args.matching,
                refinement_steps=steps,
                linear_relative_residual=linear_error,
                max_absolute_residual=residual,
                spectrum_error=spectrum,
                inverse_count=int(solver.converged.numpy()[0]),
                physical_count=int(ev.converged.numpy()[0]),
                accepted=residual <= 1e-9 and spectrum <= 1e-9,
            )
            report["cases"].append(row)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)
            solver.graph = graph = None
            gc.collect()
            inv.close()


if __name__ == "__main__":
    main()
