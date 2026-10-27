"""Accuracy-gated cost of optional fresh inverse residuals at graph finalization."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.acceptance import errors_below_tolerance
from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.side < 4 or args.repeats < 1:
        parser.error("side>=4 and repeats>=1 required")
    side, k, sigma = args.side, 8, -0.1
    one = sp.eye(side, format="csr")
    line = sp.diags([-np.ones(side - 1), np.full(side, 2.0), -np.ones(side - 1)], [-1, 0, 1])
    a = (
        sp.kron(sp.kron(line, one), one)
        + sp.kron(sp.kron(one, line), one)
        + sp.kron(sp.kron(one, one), line)
    ).tocsr()
    modes = np.arange(1, side + 1)
    eigenvalues_1d = 4 * np.sin(np.pi * modes / (2 * (side + 1))) ** 2
    triples = np.array(list(itertools.product(range(side), repeat=3)))
    eigenvalues = eigenvalues_1d[triples].sum(axis=1)
    order = np.argsort(eigenvalues)
    expected = eigenvalues[order[:k]]
    selected = triples[eigenvalues <= expected[-1] + 1e-12]
    sine = np.sqrt(2 / (side + 1)) * np.sin(np.pi * modes[:, None] * modes / (side + 1))
    allowed = np.column_stack(
        [np.kron(np.kron(sine[:, i], sine[:, j]), sine[:, q]) for i, j, q in selected]
    )
    # Analytic modes only validate outputs; they never initialize the solver.
    shifted = matrix(a - sigma * sp.eye(a.shape[0], format="csr"))
    setup_seconds, inverse = timed(lambda: CuDSSInverse(shifted, 1))
    solver = KrylovSchur(inverse, k, ncv=32, tol=1e-10)
    physical = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-9)
    fresh = EigenpairEvaluation(
        inverse, solver.eigenvectors, eigenvalues=solver.eigenvalues, rhs_width=1, tol=solver.tol
    )
    graphs, capture_times = {}, {}
    competing = set(other_gpu_processes())
    for audit in [False, True]:

        def finalize(audit=audit):
            physical.run()
            if audit:
                fresh.run()

        capture_times[audit], graphs[audit] = timed(lambda: solver.capture(12, finalizer=finalize))
    for _ in range(3):
        for graph in graphs.values():
            wp.capture_launch(graph)
    wp.synchronize_device()
    report = dict(
        args=vars(args),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        n=a.shape[0],
        nnz=a.nnz,
        k=k,
        ncv=32,
        cycles=12,
        sigma=sigma,
        setup_seconds=setup_seconds,
        setup_scope="Shared cuDSS factor/workspace setup only; fixture, solver/evaluator allocations and capture excluded",
        protocol="Dirichlet3D Poisson; independent sine-product spectrum and complete allowed space. Seed42 starts; no reference initialization. Both graphs include original Rayleigh/physical residual finalization. Optional graph additionally checks returned inverse values with8 one-RHS solves. Three warmups each then alternating timing order. Default graph receives fresh check outside timing for validation. No numerical/tolerance changes.",
        capture_seconds={str(key): value for key, value in capture_times.items()},
        rows=[],
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path("warpack/solver.py")]
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(args.repeats):
            for audit in [False, True] if repeat % 2 == 0 else [True, False]:
                competing.update(other_gpu_processes())
                seconds, _ = timed(lambda: wp.capture_launch(graphs[audit]))
                competing.update(other_gpu_processes())
                if not audit:
                    fresh.run()  # Validation only, explicitly outside timed default replay.
                v = solver.eigenvectors.numpy().T
                values = physical.values.numpy()
                inverse_values = solver.eigenvalues.numpy()
                transformed = sigma + 1 / inverse_values
                row = dict(
                    fresh_in_graph=audit,
                    repeat=repeat,
                    seconds=seconds,
                    cached_count=int(solver.converged.numpy()[0]),
                    fresh_count=int(fresh.converged.numpy()[0]),
                    physical_count=int(physical.converged.numpy()[0]),
                    status=int(solver.status.numpy()[0]),
                    fresh_residual=float(fresh.residuals.numpy().max()),
                    physical_residual=float(physical.residuals.numpy().max()),
                    returned_spectrum_error=float(abs(np.sort(transformed) - expected).max()),
                    rayleigh_spectrum_error=float(abs(np.sort(values) - expected).max()),
                    subspace_error=float(np.linalg.norm(v - allowed @ (allowed.T @ v))),
                    orthogonality_error=float(abs(v.T @ v - np.eye(k)).max()),
                )
                row["accepted"] = bool(
                    row["cached_count"] == row["fresh_count"] == row["physical_count"] == k
                    and row["status"] == 0
                    and row["fresh_residual"] <= 1e-10
                    and row["physical_residual"] <= 1e-9
                    and errors_below_tolerance(
                        [
                            row["returned_spectrum_error"],
                            row["rayleigh_spectrum_error"],
                            row["subspace_error"],
                            row["orthogonality_error"],
                        ],
                        1e-8,
                    )
                )
                report["rows"].append(row)
                out.write_text(json.dumps(report, indent=2) + "\n")
                print(audit, seconds, row["accepted"], row["fresh_count"], flush=True)
        report["graphs"] = {}
        for audit, graph in graphs.items():
            dot = Path("build") / f"{out.stem}-{audit}.dot"
            dot.parent.mkdir(exist_ok=True)
            wp.capture_debug_dot_print(graph, str(dot))
            report["graphs"][str(audit)] = graph_summary(dot)
            dot.unlink()
        report["accepted"] = all(row["accepted"] for row in report["rows"])
        report["competing_gpu_processes"] = sorted(competing)
        report["timing_eligible"] = not competing and report["accepted"]
        report["median_seconds"] = {
            str(audit): float(
                np.median([r["seconds"] for r in report["rows"] if r["fresh_in_graph"] == audit])
            )
            for audit in [False, True]
        }
        out.write_text(json.dumps(report, indent=2) + "\n")
        if not report["timing_eligible"]:
            raise SystemExit("Accuracy/occupancy gate rejected; all evidence retained")
    finally:
        solver.graph = None
        graphs.clear()
        wp.synchronize_device()
        inverse.close()


if __name__ == "__main__":
    main()
