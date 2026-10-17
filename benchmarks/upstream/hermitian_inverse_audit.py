"""Small correctness-only controls for factor matching and Hermitian iteration."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from warpack import CuDSSInverse, ShiftInvertEigensolver

from .adapters import complex_matrix, output


def run(a, k, ncv, hermitian, matching):
    solver = ShiftInvertEigensolver(complex_matrix(a), k, ncv=ncv, tol=1e-12, hermitian=hermitian)
    graph = None
    try:
        if not matching:
            # Controlled reproduction of the old factor configuration. Do this
            # before warmup/capture; no graph references the replaced factors.
            solver.inverse.inverse.close()
            solver.inverse.inverse = CuDSSInverse(
                solver.inverse.matrix, 1, mtype="general", matching=False, refinement_steps=1
            )
        real = np.block([[a.real, -a.imag], [a.imag, a.real]])
        matrix = solver.inverse.matrix
        assembled = sp.csr_matrix(
            (matrix.values.numpy(), matrix.columns.numpy(), matrix.offsets.numpy()),
            shape=matrix.shape,
        ).toarray()
        b = np.vstack([np.ones(len(real)), np.random.default_rng(17).normal(size=len(real))])
        rhs = wp.array(b, dtype=wp.float64)
        solution = wp.empty_like(rhs)
        solver.inverse.apply(rhs, solution)
        inverse_residual = np.max(abs(solution.numpy() @ real.T - b))
        graph = solver.capture(30)
        wp.capture_launch(graph)
        values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
        ref = np.linalg.eigvalsh(a)
        ref = ref[np.argsort(abs(ref))[:k]]
        return dict(
            n=len(a),
            k=k,
            ncv=ncv,
            hermitian=hermitian,
            matching=matching,
            refinement_steps=1,
            iteration_budget=30,
            assembly_error=float(np.max(abs(assembled - real))),
            inverse_max_absolute_residual=float(inverse_residual),
            max_absolute_residual=float(np.max(abs(a @ vectors - vectors * values))),
            spectrum_error=float(np.max(abs(np.sort(values) - np.sort(ref)))),
            orthogonality_error=float(np.max(abs(vectors.conj().T @ vectors - np.eye(k)))),
            reported_converged=int(solver.converged.numpy()[0]),
        )
    finally:
        graph = solver.graph = None
        (solver.core.real if hermitian else solver.core).graph = None
        solver.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/spectra_hermitian_sparse.npz"
    data = np.load(fixture)
    report = dict(protocol="Small accuracy controls only; no timings", cases=[])
    for n, k, ncv in [(10, 3, 6), (100, 10, 20)]:
        for matching in [False, True]:
            for hermitian in [False, True]:
                row = run(data[f"a{n}"], k, ncv, hermitian, matching)
                report["cases"].append(row)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
