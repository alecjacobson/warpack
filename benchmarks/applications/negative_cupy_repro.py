"""Isolate long-iteration CuPy LOBPCG behavior without Warp/cuDSS computation."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
import warnings
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import numpy as np

from benchmarks.acceptance import errors_below_tolerance

from .negative_foundation import make_negative_foundation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=24)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    case = make_negative_foundation(args.side, args.modes, "LM", consistent=False)
    p = case.problem
    mass = float(p.b.diagonal()[0])
    assert np.all(p.b.diagonal() == mass) and p.b.nnz == p.n
    digest = hashlib.sha256()
    for matrix in (p.a, p.b):
        for values in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(values.tobytes())
    report = dict(
        args=vars(args),
        n=p.n,
        k=p.k,
        mass_scalar=mass,
        matrix_sha256=digest.hexdigest(),
        cupy=importlib.metadata.version("cupy-cuda12x"),
        native_source_sha256=hashlib.sha256(
            Path(inspect.getfile(cl.lobpcg)).read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warpack_loaded="warpack" in sys.modules,
        scope="Native CuPy only: no Warp kernels, cuDSS factors, interop streams or graph capture. Analytic fixture/reference and validation on host. No timing claims.",
        trials=[],
    )
    assert not report["warpack_loaded"]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    initial = np.random.default_rng(2026).normal(size=(p.n, p.k))
    for representation, iterations in [
        ("generalized_exact_scalar_preconditioner", 200),
        ("generalized_exact_scalar_preconditioner", 1000),
        ("generalized_identity_preconditioner", 1000),
        ("scaled_standard", 1000),
    ]:
        standard = representation == "scaled_standard"
        a = cs.csr_matrix(p.a / mass if standard else p.a)
        b = None if standard else cs.csr_matrix(p.b)
        preconditioner = None
        if representation == "generalized_exact_scalar_preconditioner":
            preconditioner = cl.LinearOperator(
                (p.n, p.n), matvec=lambda x: x / mass, matmat=lambda x: x / mass, dtype=cp.float64
            )
        for trial in range(args.repeats):
            cp.random.seed(2026)
            x = cp.asarray(initial * (np.sqrt(mass) if standard else 1))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                w, v, history = cl.lobpcg(
                    a,
                    x,
                    B=b,
                    M=preconditioner,
                    largest=True,
                    tol=1e-9,
                    maxiter=iterations,
                    retResidualNormsHistory=True,
                )
            w, v = cp.asnumpy(w), cp.asnumpy(v)
            if standard:
                v /= np.sqrt(mass)
            av, bv = p.a @ v, p.b @ v
            residuals = np.linalg.norm(av - bv * w, axis=0)
            scaled = residuals / np.maximum(
                1, np.linalg.norm(av, axis=0) + abs(w) * np.linalg.norm(bv, axis=0)
            )
            orthogonality = float(abs(v.T @ bv - np.eye(p.k)).max())
            remainder = v - p.requested_basis @ (p.requested_basis.T @ bv)
            subspace = float(np.sqrt(max(0, np.sum(remainder * (p.b @ remainder)))))
            spectrum = float(
                np.max(
                    abs(np.sort(w) - np.sort(p.expected)) / np.maximum(1, abs(np.sort(p.expected)))
                )
            )
            native_residuals = residuals / np.sqrt(mass) if standard else residuals
            row = dict(
                representation=representation,
                maxiter=iterations,
                trial=trial,
                residual=float(scaled.max()),
                absolute_residual=float(residuals.max()),
                native_absolute_residual=float(native_residuals.max()),
                native_stopping_passed=bool(np.all(native_residuals <= 1e-9)),
                orthogonality=orthogonality,
                requested_subspace_error=subspace,
                spectrum_error=spectrum,
                eigenvalues=w.tolist(),
                physical_accepted=bool(
                    errors_below_tolerance([scaled.max(), orthogonality, subspace, spectrum], 1e-8)
                ),
                history_maxima=[float(cp.asnumpy(item).max()) for item in history],
                warnings=[str(item.message) for item in caught],
                input_matrix_unchanged=bool(
                    cp.asnumpy(a.data).tobytes() == (p.a / mass if standard else p.a).data.tobytes()
                ),
            )
            report["trials"].append(row)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(
                representation,
                iterations,
                trial,
                row["physical_accepted"],
                row["native_absolute_residual"],
                row["orthogonality"],
                flush=True,
            )


if __name__ == "__main__":
    main()
