"""Invariant warm starts: distinguish residual convergence from target coverage."""

import argparse
import json
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sl
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator


def quality(a, values, vectors, k):
    order = np.argsort(values)
    values, vectors = values[order][:k], vectors[:, order][:, :k]
    av = a @ vectors
    absolute = np.linalg.norm(av - vectors * values, axis=0)
    residuals = absolute / np.maximum(1, np.linalg.norm(av, axis=0) + abs(values))
    reference, basis = la.eigh(a)
    wanted = basis[:, :k]
    error = float(abs(values - reference[:k]).max())
    subspace = float(la.norm(vectors - wanted @ (wanted.T @ vectors)))
    return dict(
        values=values.tolist(),
        absolute_residuals=absolute.tolist(),
        normalized_residuals=residuals.tolist(),
        requested_eigenvalue_error=error,
        requested_subspace_error=subspace,
        orthogonality=float(la.norm(vectors.T @ vectors - np.eye(k))),
        requested_spectrum_pass=error < 1e-8 and subspace < 1e-6,
    )


def warp_solve(a, start, k, budget, noise, pre=None):
    solver = BlockLOBPCG(
        SparseOperator(matrix(a)),
        k,
        width=start.shape[0],
        initial_vectors=wp.array(start, dtype=wp.float64),
        initial_perturbation=noise,
        preconditioner=None if pre is None else SparseOperator(matrix(pre)),
        tol=1e-10,
    )
    wp.capture_launch(solver.capture(budget))
    result = quality(a, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, k)
    result.update(
        backend="warp",
        initial_perturbation=noise,
        budget=budget,
        status=int(solver.status.numpy()[0]),
        converged=int(solver.converged.numpy()[0]),
        iterations=int(solver.iterations.numpy()[0]),
    )
    return result


def native_solve(a, start, k, backend, pre=None):
    if backend == "cupy":
        values, vectors = cl.lobpcg(
            cs.csr_matrix(cp.asarray(a)),
            cp.asarray(start.T.copy()),
            M=None if pre is None else cs.csr_matrix(cp.asarray(pre)),
            largest=False,
            tol=1e-10,
            maxiter=100,
        )
        values, vectors = values.get(), vectors.get()
    else:
        values, vectors = sl.lobpcg(a, start.T.copy(), M=pre, largest=False, tol=1e-10, maxiter=100)
    return dict(
        backend=backend, initial_perturbation=0.0, budget=100, **quality(a, values, vectors, k)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    n, width, k = 64, 8, 4
    old = np.arange(1.0, n + 1)
    a = np.diag(old)
    source = BlockLOBPCG(
        SparseOperator(matrix(a)),
        width,
        preconditioner=SparseOperator(matrix(np.diag(1 / (old + 1)))),
        tol=1e-12,
    )
    wp.capture_launch(source.capture(100))
    assert source.status.numpy()[0] == 0 and source.converged.numpy()[0] == width
    new = old.copy()
    new[:width] += 100
    a, pre = np.diag(new), np.diag(1 / (new + 1))
    fixtures = []
    for kind, start in (
        ("exact_old_basis", np.eye(n)[:width].copy()),
        ("computed_old_block", source.x.numpy().copy()),
    ):
        rows = [warp_solve(a, start, k, 100, noise, pre) for noise in (0.0, 1e-3)]
        rows += [native_solve(a, start, k, backend, pre) for backend in ("scipy", "cupy")]
        assert rows[1]["requested_spectrum_pass"] and rows[1]["converged"] == k
        fixtures.append(dict(name=kind, n=n, k=k, width=width, rows=rows))
    n, k = 12, 2
    adjacency = la.toeplitz(np.r_[0.0, 1.0, np.zeros(n - 2)])
    a = np.diag(adjacency.sum(axis=1)) - adjacency
    start = np.cos(np.outer(np.arange(n) + 0.5, np.pi * np.arange(n) / n))[:, -k:].T.copy()
    rows = [warp_solve(a, start, k, budget, noise) for noise in (0.0, 1e-3) for budget in (20, 100)]
    rows += [native_solve(a, start, k, backend) for backend in ("scipy", "cupy")]
    assert rows[3]["requested_spectrum_pass"] and rows[3]["converged"] == k
    fixtures.append(dict(name="upstream_fiedler_high_start", n=n, k=k, width=k, rows=rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                versions=dict(warp=wp.__version__, cupy=cp.__version__),
                fixtures=fixtures,
                policy="No target reference vectors enter starts. Only Warp's explicit exploration rows add noise; native rows are unperturbed controls. Reference eigensolves validate output only. No timing claims.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
