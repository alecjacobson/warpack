"""Reciprocal force-prior comparison; both covariance and Q residuals are required."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.graph_audit import graph_summary
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import matrix

from .fem_hierarchy import beam_problem, matrix_hash
from .fem_shapes import elapsed
from .force_covariance_compare import quality, reference
from .force_reciprocal import ForceReciprocalSolver


def reciprocal_residual(q, mass, u, mu):
    au, bu = q @ u, mass @ u
    return float(
        max(
            np.linalg.norm(au - bu * mu, axis=0)
            / np.maximum(1, np.linalg.norm(au, axis=0) + abs(mu) * np.linalg.norm(bu, axis=0))
        )
    )


class ExplicitReciprocal:
    """Independent CPU implementation of the reciprocal formulation, SuperLU backend."""

    def __init__(self, h, mass, diagonal):
        self.q = (h @ sp.diags(1 / diagonal) @ h).tocsc()
        self.mass = mass
        self.factor = sla.splu(self.q)
        self.inverse = sla.LinearOperator(h.shape, matvec=self.factor.solve, dtype=np.float64)

    def solve(self, k, ncv, tol):
        mu, u = sla.eigsh(
            self.q,
            M=self.mass,
            k=k,
            ncv=min(ncv, self.q.shape[0]),
            sigma=0,
            which="LM",
            OPinv=self.inverse,
            tol=tol,
            maxiter=10000,
            v0=np.random.default_rng(915).normal(size=self.q.shape[0]),
        )
        order = np.argsort(mu)
        return mu[order], u[:, order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--mass", choices=("lumped", "consistent"), default="lumped")
    parser.add_argument("--backend", choices=("arpack", "cg", "cudss"), required=True)
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--fixed-cycles", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--refinement-steps", type=int, default=0)
    parser.add_argument("--polish-steps", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.side, args.modes, args.cycles, args.repeats) < 1
        or args.ncv <= args.modes
        or not np.isfinite(args.tol)
        or not 0 < args.tol <= 1e-8
        or (args.backend == "arpack" and args.fixed_cycles)
        or args.refinement_steps < 0
        or (args.backend != "cg" and args.refinement_steps != 0)
        or args.polish_steps < 0
        or (args.backend == "arpack" and args.polish_steps != 0)
    ):
        parser.error("positive sizes, ncv>modes, 0<tol<=1e-8 and Warp-only fixed cycles required")
    start = time.perf_counter()
    p = beam_problem(args.side, args.modes, args.mass == "consistent")
    diagonal = p.b.diagonal() * np.linspace(0.2, 2, p.n)
    prior, prior_inverse = sp.diags(diagonal, format="csr"), sp.diags(1 / diagonal, format="csr")
    assembly_seconds = time.perf_counter() - start
    start = time.perf_counter()
    host, ref = reference(p, prior, "diagonal")
    q_reference = p.a @ prior_inverse @ p.a
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    driver = None
    if args.backend == "arpack":
        measured = ExplicitReciprocal(p.a, p.b, diagonal)
    else:
        driver = ForceReciprocalSolver(
            matrix(p.a),
            matrix(p.b),
            matrix(prior),
            matrix(prior_inverse),
            p.k,
            args.backend,
            ncv=args.ncv,
            tol=args.tol,
            refinement_steps=args.refinement_steps,
            polish_steps=args.polish_steps,
        )
        wp.synchronize()
    setup_seconds = time.perf_counter() - start
    capture_seconds = (
        elapsed(lambda: driver.capture(args.cycles, adaptive=False if args.fixed_cycles else None))
        if driver
        else 0.0
    )
    rows = []
    try:
        for trial in range(args.repeats + 1):
            if driver:
                seconds = elapsed(driver.solve)
                mu = driver.solver.eigenvalues.numpy()
                values, u = driver.eigenvalues.numpy(), driver.eigenvectors.numpy().T
            else:
                start = time.perf_counter()
                mu, u = measured.solve(p.k, args.ncv, args.tol)
                values = 1 / mu
                seconds = time.perf_counter() - start
            checks = quality(
                values,
                u,
                host,
                ref,
                SimpleNamespace(solver=driver.solver, inverses=[driver.inverse])
                if driver
                else None,
            )
            checks["reciprocal_residual"] = reciprocal_residual(q_reference, p.b, u, mu)
            checks["accepted"] &= bool(
                np.isfinite(checks["reciprocal_residual"]) and checks["reciprocal_residual"] < 1e-8
            )
            row = dict(
                seconds=seconds,
                quality=checks,
                iterations=None if driver is None else int(driver.krylov.iterations.numpy()[0]),
                recoveries=None if driver is None else int(driver.krylov.recoveries.numpy()[0]),
                krylov_physical_converged=None
                if driver is None
                else int(driver.krylov.converged.numpy()[0]),
                polish_iterations=None
                if driver is None or driver.polisher is None
                else int(driver.polisher.iterations.numpy()[0]),
            )
            row["exploration_guard_met"] = (
                None
                if driver is None
                else bool(driver.krylov.m == driver.krylov.n or row["recoveries"] + 1 >= p.k)
            )
            row["accepted"] = bool(
                checks["accepted"]
                and (driver is None or not driver.adaptive or row["exploration_guard_met"])
            )
            print(trial, row, flush=True)
            if trial:
                rows.append(row)
            else:
                warmup = row
        args.out.parent.mkdir(parents=True, exist_ok=True)
        graph = None
        if driver:
            stages = []
            for name, stage in (
                ("solve", driver.graph),
                ("polish", driver.polish_graph),
                ("recover", driver.recovery_graph),
            ):
                if stage is None:
                    continue
                dot = args.out.with_suffix(f".{name}.dot")
                wp.capture_debug_dot_print(stage, str(dot))
                stages.append(graph_summary(dot))
                dot.unlink()
            graph = {
                key: dict(sum((Counter(s[key]) for s in stages), Counter()))
                for key in ("node_types_including_conditional_body", "memcpy_kinds")
            }
            graph["stages"] = stages
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert not any(
                "DtoH" in k or (args.backend == "cg" and "HtoD" in k) for k in graph["memcpy_kinds"]
            )
        digest = hashlib.sha256()
        for array in (prior.indptr, prior.indices, prior.data):
            digest.update(array.tobytes())
        report = dict(
            formulation="reciprocal",
            arguments=vars(args) | dict(out=str(args.out)),
            n=p.n,
            outer_capture="arpack"
            if driver is None
            else "adaptive"
            if driver.adaptive
            else "fixed",
            prior_rank_bound=p.n,
            reference=ref["method"],
            guard=ref["guard"],
            assembly_seconds=assembly_seconds,
            setup_seconds=setup_seconds,
            capture_seconds=capture_seconds,
            reference_seconds=reference_seconds,
            stiffness_nnz=p.a.nnz,
            reciprocal_nnz=q_reference.nnz,
            cpu_factor_nnz=measured.factor.L.nnz + measured.factor.U.nnz
            if driver is None
            else None,
            warmup=warmup,
            rows=rows,
            graph=graph,
            matrix_sha256=matrix_hash(p),
            prior_sha256=digest.hexdigest(),
            source_sha256=source_manifest(__file__),
            dependency=warp_dependencies(),
            accepted=all(r["accepted"] for r in [warmup] + rows),
        )
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if not report["accepted"]:
            raise SystemExit("Rejected reciprocal solve; evidence retained")
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
