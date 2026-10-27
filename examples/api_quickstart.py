"""Small frontend examples using Warp only, with optional cuDSS.

Run from a source checkout:
    python -m examples.api_quickstart
    python -m examples.api_quickstart --cudss

Each result is checked against a known diagonal spectrum. Host inspection is
explicit and outside captured computation. No meshes or reference packages
are needed. See docs/api/README.md for complete API contracts.
"""

import argparse
import math

import warp as wp
import warp.sparse as sparse
from warp.optim import linear

from warpack import (
    BlockLOBPCG,
    CGInverse,
    CuDSSInverse,
    GeneralizedKrylovSchur,
    HermitianCSR,
    HermitianEigensolver,
    KrylovSchur,
    PartialSVD,
    SparseOperator,
)


def diagonal(values, nrow=None):
    """Assemble a compact scalar FP64 matrix on the current Warp device."""
    n = len(values)
    indices = wp.array(list(range(n)), dtype=wp.int32)
    return sparse.bsr_from_triplets(
        n if nrow is None else nrow,
        n,
        indices,
        indices,
        wp.array(values, dtype=wp.float64),
    )


def replay(solver, budget):
    """Complete the first upload during setup, then replay on the same stream."""
    graph = solver.capture(budget)
    wp.capture_launch(graph)
    wp.synchronize_device(solver.device)
    wp.capture_launch(graph)
    wp.synchronize_device(solver.device)


def check(label, values, expected, status, *counts):
    """Explicit example-only host acceptance, after graph execution."""
    if int(status.numpy()[0]) != 0 or any(int(c.numpy()[0]) != k for c, k in counts):
        raise RuntimeError(f"{label}: incomplete or failed solve")
    actual = sorted(float(x) for x in values.numpy())
    if not all(
        math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)
        for a, b in zip(actual, sorted(expected), strict=True)
    ):
        raise RuntimeError(f"{label}: incorrect requested eigenvalues: {actual}")
    print(f"{label}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cudss", action="store_true", help="also check a cuDSS generalized solve")
    args = parser.parse_args()
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        n, k = 64, 4
        stiffness = [float(i + 1) for i in range(n)]
        mass = [1.0 + 0.01 * i for i in range(n)]
        K, M = diagonal(stiffness), diagonal(mass)
        low = [stiffness[i] / mass[i] for i in range(k)]

        standard = KrylovSchur(SparseOperator(K), k, ncv=24, which="LA", tol=1e-10)
        replay(standard, 50)
        check(
            "KrylovSchur",
            standard.eigenvalues,
            stiffness[-k:],
            standard.status,
            (standard.converged, k),
        )

        block = BlockLOBPCG(
            SparseOperator(K),
            k,
            width=8,
            mass=SparseOperator(M),
            preconditioner=linear.preconditioner(K, ptype="diag"),
            tol=1e-9,
        )
        replay(block, 200)
        check("BlockLOBPCG", block.eigenvalues, low, block.status, (block.converged, k))

        sigma = -1.0
        shifted = diagonal([a - sigma * b for a, b in zip(stiffness, mass, strict=True)])
        backends = ["cg", "cudss"] if args.cudss else ["cg"]
        for backend in backends:
            inverse = (
                CuDSSInverse(shifted, width=1, mtype="spd")
                if backend == "cudss"
                else CGInverse(shifted, preconditioner="diag", tol=1e-14, atol=0.0)
            )
            solver = GeneralizedKrylovSchur(
                SparseOperator(K),
                SparseOperator(M),
                inverse,
                k,
                ncv=24,
                target=sigma,
                tol=1e-10,
            )
            try:
                replay(solver, 20)
                check(
                    f"GeneralizedKrylovSchur + {backend}",
                    solver.eigenvalues,
                    low,
                    solver.status,
                    (solver.converged, k),
                )
            finally:
                # No graph is still executing. Drop captures before releasing factors.
                solver.graph = None
                if backend == "cudss":
                    inverse.close()

        h = HermitianCSR(
            n,
            wp.array(list(range(n + 1)), dtype=wp.int32),
            wp.array(list(range(n)), dtype=wp.int32),
            wp.array([[value, 0.0] for value in stiffness], dtype=wp.vec2d),
        )
        hermitian = HermitianEigensolver(h, k, ncv=48, which="LA", tol=1e-10)
        replay(hermitian, 50)
        check(
            "HermitianEigensolver",
            hermitian.eigenvalues,
            stiffness[-k:],
            hermitian.real.status,
            (hermitian.real.converged, 2 * k),
            (hermitian.count, k),
        )
        # Complex residuals are checked independently on the host for this diagonal example.
        for value, vector in zip(
            hermitian.eigenvalues.numpy(), hermitian.eigenvectors.numpy(), strict=True
        ):
            error = math.sqrt(
                sum(
                    (a - value) ** 2 * (z[0] ** 2 + z[1] ** 2)
                    for a, z in zip(stiffness, vector, strict=True)
                )
            )
            if error / max(1.0, abs(value)) > 1e-10:
                raise RuntimeError("HermitianEigensolver: complex residual failed")

        singular = [float(i + 1) for i in range(32)]
        svd = PartialSVD(diagonal(singular, nrow=64), k, ncv=48, tol=1e-10)
        replay(svd, 50)
        check(
            "PartialSVD",
            svd.singular_values,
            singular[-k:],
            svd.status,
            (svd.count_left, k),
            (svd.count_right, k),
        )


if __name__ == "__main__":
    main()
