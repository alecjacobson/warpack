"""Preserve strict-count variability of the two-cycle small Hodge profile."""

import json
from pathlib import Path

import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

from .hodge import make_hodge_case


def main():
    p = make_hodge_case("large_disconnected", side=7).problem
    rows = []
    for refinement in (0, 1):
        for cycles in (2, 3, 4):
            inverse = CuDSSInverse(matrix(p.a - p.target * p.b), 1, refinement_steps=refinement)
            solver = GeneralizedKrylovSchur(
                SparseOperator(matrix(p.a)),
                SparseOperator(matrix(p.b)),
                inverse,
                p.k,
                ncv=32,
                target=p.target,
                tol=1e-11,
            )
            graph = solver.capture(cycles)
            checks = []
            try:
                for _ in range(20):
                    wp.capture_launch(graph)
                    checks.append(
                        dict(
                            count=int(solver.converged.numpy()[0]),
                            residuals=solver.residuals.numpy().tolist(),
                        )
                    )
                rows.append(dict(refinement=refinement, cycles=cycles, checks=checks))
                print(refinement, cycles, min(c["count"] for c in checks), flush=True)
            finally:
                solver.graph = graph = None
                inverse.close()
    Path("results/hodge_strict_probe.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
