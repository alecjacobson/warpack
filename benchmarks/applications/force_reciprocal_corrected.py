"""Standalone CPU reciprocal comparison with explicitly charged H-based recovery.

This is a benchmark adapter, not native ARPACK or installed Warpack code. It
constructs its own Q and H factors; the independent reference lends neither.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from benchmarks.anvil_inputs import array_identity, matrix_identity
from benchmarks.source_provenance import source_manifest

from .fem_hierarchy import beam_problem
from .force_covariance_compare import HostCovariance, quality, reference
from .force_reciprocal_compare import ExplicitReciprocal, reciprocal_residual
from .force_reciprocal_conditioning import covariance_ritz


class CorrectedReciprocal:
    """Explicit-Q ARPACK plus one inverse step and covariance Ritz recovery."""

    def __init__(self, h, mass, diagonal):
        start = time.perf_counter()
        self.explicit = ExplicitReciprocal(h, mass, diagonal)
        self.q_setup_seconds = time.perf_counter() - start
        start = time.perf_counter()
        self.mass, self.prior = mass, sp.diags(diagonal, format="csr")
        self.h_factor = sla.splu(h.tocsc())
        self.recovery_setup_seconds = time.perf_counter() - start

    def apply(self, x):
        return self.mass @ self.h_factor.solve(self.prior @ self.h_factor.solve(self.mass @ x))

    def solve(self, k, ncv, tol=1e-12):
        start = time.perf_counter()
        mu, basis = self.explicit.solve(k, ncv, tol)
        native_seconds = time.perf_counter() - start
        start = time.perf_counter()
        values, vectors = covariance_ritz(self, basis, inverse_step=True)
        recovery_seconds = time.perf_counter() - start
        return (
            values,
            vectors,
            dict(
                native_seconds=native_seconds,
                recovery_seconds=recovery_seconds,
                native_values=1 / mu,
                native_basis=basis,
            ),
        )


def checks(values, basis, host, ref, q):
    result = quality(values, basis, host, ref)
    result["reciprocal_residual"] = reciprocal_residual(q, host.mass, basis, 1 / values)
    result["accepted"] &= bool(
        np.isfinite(result["reciprocal_residual"]) and result["reciprocal_residual"] < 1e-8
    )
    return result


def run(side=10, modes=20, consistent=False, backend="corrected", repeats=3):
    start = time.perf_counter()
    problem = beam_problem(side, modes, consistent)
    diagonal = problem.b.diagonal() * np.linspace(0.2, 2, problem.n)
    prior = sp.diags(diagonal, format="csr")
    assembly_seconds = time.perf_counter() - start
    start = time.perf_counter()
    host, ref = reference(problem, prior, "diagonal")
    q = (problem.a @ sp.diags(1 / diagonal) @ problem.a).tocsc()
    reference_seconds = time.perf_counter() - start
    # References are complete before measured setup; construct independent factors.
    start = time.perf_counter()
    if backend == "corrected":
        measured = CorrectedReciprocal(problem.a, problem.b, diagonal)
    elif backend == "implicit":
        measured = HostCovariance(problem.a, problem.b, prior)
    else:
        raise ValueError("expected corrected or implicit backend")
    setup_seconds = time.perf_counter() - start
    rows = []
    for trial in range(repeats + 1):
        start = time.perf_counter()
        if backend == "corrected":
            values, basis, stages = measured.solve(problem.k, max(64, 2 * problem.k + 1))
        else:
            values, basis = measured.solve(problem.k, max(64, 2 * problem.k + 1))
            stages = {}
        seconds = time.perf_counter() - start
        native_quality = (
            checks(stages.pop("native_values"), stages.pop("native_basis"), host, ref, q)
            if backend == "corrected"
            else None
        )
        physical = checks(values, basis, host, ref, q)
        row = dict(
            seconds=seconds,
            stages=stages,
            quality=physical,
            native_quality=native_quality,
            accepted=physical["accepted"],
        )
        print(trial, row, flush=True)
        if trial:
            rows.append(row)
        else:
            warmup = row
    return dict(
        backend=backend,
        n=problem.n,
        k=problem.k,
        inputs=dict(
            h=matrix_identity(problem.a),
            mass=matrix_identity(problem.b),
            prior=matrix_identity(prior),
            q=matrix_identity(q),
            reference_values=array_identity(ref["values"]),
            reference_basis=array_identity(ref["basis"]),
            guard=ref["guard"],
            initial=array_identity(np.random.default_rng(915).normal(size=problem.n)),
        ),
        assembly_seconds=assembly_seconds,
        reference_seconds=reference_seconds,
        setup_seconds=setup_seconds,
        setup_stages=None
        if backend == "implicit"
        else dict(q=measured.q_setup_seconds, recovery=measured.recovery_setup_seconds),
        factor_nnz=dict(
            h=measured.h_factor.L.nnz + measured.h_factor.U.nnz,
            q=measured.explicit.factor.L.nnz + measured.explicit.factor.U.nnz
            if backend == "corrected"
            else None,
            mass=measured.m_factor.L.nnz + measured.m_factor.U.nnz
            if backend == "implicit"
            else None,
        ),
        warmup=warmup,
        rows=rows,
        accepted=all(row["accepted"] for row in [warmup] + rows),
        versions=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__
        ),
        source_sha256=source_manifest(__file__),
        protocol="CPU-only comparison, one separately validated warmup and measured repeats. Every solve requires both original covariance and reciprocal residuals plus all existing spectrum/subspace/orthogonality/count gates at 1e-8. Corrected is explicit-Q ARPACK/SuperLU plus an H-based inverse step and covariance Ritz rotation, not unmodified ARPACK. Its setup includes independent Q assembly/factor and H factor; recovery includes four H block solves (two inverse-step, two projection), sparse products and dense Ritz work. Implicit independently constructs H/M factors. No reference factor reuse. Assembly/reference/validation are excluded from solve timing and setup scope is separate. Native pre-correction failures are retained.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--consistent", action="store_true")
    parser.add_argument("--backend", choices=("corrected", "implicit"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.side, args.modes, args.repeats) < 1:
        parser.error("positive mesh size, mode count and repeat count required")
    if args.out.exists():
        parser.error("refusing to overwrite an existing report")
    report = run(args.side, args.modes, args.consistent, args.backend, args.repeats)
    report["arguments"] = vars(args) | dict(out=str(args.out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Rejected CPU comparison; evidence retained")


if __name__ == "__main__":
    main()
