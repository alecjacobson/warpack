"""Implicit force-prior ROM comparison with independent physical references."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import matrix

from .fem_hierarchy import beam_problem, matrix_hash
from .fem_shapes import elapsed
from .force_covariance import ForceCovarianceSolver
from .force_dual import localized_forces
from .force_dual import reference as snapshot_reference


class HostCovariance:
    def __init__(self, h, mass, prior):
        self.mass, self.prior = mass, prior
        self.h_factor = sla.splu(h.tocsc())
        self.m_factor = sla.splu(mass.tocsc())
        self.operator = sla.LinearOperator(
            h.shape, matvec=self.apply, matmat=self.apply, dtype=np.float64
        )
        self.minv = sla.LinearOperator(h.shape, matvec=self.m_factor.solve, dtype=np.float64)

    def apply(self, x):
        return self.mass @ self.h_factor.solve(self.prior @ self.h_factor.solve(self.mass @ x))

    def solve(self, k, ncv, seed=915):
        values, basis = sla.eigsh(
            self.operator,
            k=k,
            M=self.mass,
            Minv=self.minv,
            which="LA",
            ncv=min(self.mass.shape[0], ncv),
            tol=1e-12,
            maxiter=10000,
            v0=np.random.default_rng(seed).normal(size=self.mass.shape[0]),
        )
        return values[::-1].copy(), basis[:, ::-1].copy()


def reference(p, prior, kind, forces=None):
    host = HostCovariance(p.a, p.b, prior)
    if kind == "mass":
        inverse = sla.LinearOperator(p.a.shape, matvec=host.h_factor.solve, dtype=np.float64)
        lam, basis = sla.eigsh(
            p.a,
            k=p.k + 1,
            M=p.b,
            sigma=0,
            OPinv=inverse,
            which="LM",
            tol=1e-12,
            ncv=min(p.n, max(64, 4 * p.k)),
            v0=np.random.default_rng(913).normal(size=p.n),
        )
        order = np.argsort(lam)
        values = 1 / lam[order] ** 2
        basis = basis[:, order]
        method = "Independent stiffness/mass ARPACK; covariance values are inverse squared elastic values"
    elif kind == "localized":
        ref = snapshot_reference(
            p.a, p.b, forces, p.k + 1, method="svd" if p.b.nnz == p.n else "gram"
        )
        values, basis = ref["values"], ref["basis"]
        method = "Independent force-response snapshot reduction"
    else:
        values, basis = host.solve(p.k + 1, max(96, 4 * p.k), seed=913)
        method = "Independent expanded ARPACK implicit covariance reference"
    if (
        basis is None
        or values[p.k - 1] <= 0
        or values[p.k - 1] - values[p.k]
        <= 1e-7 * max(abs(values[p.k - 1]), abs(values[p.k]), np.finfo(float).tiny)
    ):
        raise ValueError("insufficient rank or reference boundary splits a cluster")
    return host, dict(
        values=values[: p.k], basis=basis[:, : p.k], method=method, guard=float(values[p.k])
    )


def quality(values, u, host, ref, driver=None):
    mu, au = host.mass @ u, host.apply(u)
    k = len(values)
    remainder = u - ref["basis"] @ (ref["basis"].T @ mu)
    result = dict(
        residual=float(
            np.max(
                np.linalg.norm(au - mu * values, axis=0)
                / np.maximum(
                    1, np.linalg.norm(au, axis=0) + abs(values) * np.linalg.norm(mu, axis=0)
                )
            )
        ),
        orthogonality=float(np.max(abs(u.T @ mu - np.eye(k)))),
        spectrum_error=float(
            np.max(abs(values - ref["values"]) / np.maximum(1, abs(ref["values"])))
        ),
        subspace_error=float(np.sqrt(max(0, np.sum(remainder * (host.mass @ remainder))))),
        captured_energy_error=float(
            abs(np.trace(u.T @ au) - sum(ref["values"])) / sum(ref["values"])
        ),
        positive=int(np.count_nonzero(values > 1e-12 * values[0])),
        status=0 if driver is None else int(driver.solver.status.numpy()[0]),
        converged=k if driver is None else int(driver.solver.converged.numpy()[0]),
        inner=[],
    )
    if driver is not None:
        for inv in driver.inverses:
            result["inner"].append(
                None
                if not hasattr(inv, "stats")
                else dict(
                    native=inv.stats.numpy().tolist(),
                    fresh=inv.fresh_stats.numpy().tolist(),
                    maxima=inv.residual_maxima.numpy().tolist(),
                )
            )
    result["accepted"] = bool(
        result["status"] == 0
        and result["converged"] == k
        and result["positive"] == k
        and all(
            np.isfinite(result[key]) and 0 <= result[key] < 1e-8
            for key in (
                "residual",
                "orthogonality",
                "spectrum_error",
                "subspace_error",
                "captured_energy_error",
            )
        )
        and all(
            inner is None
            or (
                inner["native"][2] == 0
                and np.isfinite(inner["maxima"][1])
                and inner["maxima"][1] < 1e-8
            )
            for inner in result["inner"]
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--mass", choices=("lumped", "consistent"), default="lumped")
    parser.add_argument("--prior", choices=("mass", "diagonal", "localized"), default="mass")
    parser.add_argument("--backend", choices=("cudss", "cg", "arpack"), required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--fixed-cycles", action="store_true")
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.fixed_cycles and args.backend == "arpack":
        parser.error("--fixed-cycles configures Warp graph capture only")
    if (
        min(args.side, args.modes, args.cycles, args.repeats) < 1
        or args.ncv <= args.modes
        or args.columns <= args.modes
    ):
        parser.error("positive sizes, ncv>modes and columns>modes required")
    start = time.perf_counter()
    p = beam_problem(args.side, args.modes, args.mass == "consistent")
    forces = None
    if args.prior == "mass":
        prior = p.b.copy()
    elif args.prior == "diagonal":
        prior = sp.diags(p.b.diagonal() * np.linspace(0.2, 2, p.n), format="csr")
    else:
        forces = localized_forces(args.side, args.columns)
        prior = sp.diags(np.sum(forces**2, axis=1), format="csr")
    assembly_seconds = time.perf_counter() - start
    start = time.perf_counter()
    host, ref = reference(p, prior, args.prior, forces)
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    driver = None
    if args.backend == "arpack":
        measured = HostCovariance(p.a, p.b, prior)
    else:
        driver = ForceCovarianceSolver(
            matrix(p.a), matrix(p.b), matrix(prior), p.k, args.backend, ncv=args.ncv
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
                values, u = driver.solver.eigenvalues.numpy(), driver.solver.eigenvectors.numpy().T
            else:
                start = time.perf_counter()
                values, u = measured.solve(p.k, args.ncv)
                seconds = time.perf_counter() - start
            row = dict(
                seconds=seconds,
                iterations=None if driver is None else int(driver.solver.iterations.numpy()[0]),
                quality=quality(values, u, host, ref, driver),
            )
            row["recoveries"] = None if driver is None else int(driver.solver.recoveries.numpy()[0])
            row["exploration_guard_met"] = (
                None
                if driver is None
                else bool(driver.solver.m == driver.solver.n or row["recoveries"] + 1 >= p.k)
            )
            row["accepted"] = bool(
                row["quality"]["accepted"]
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
            dot = args.out.with_suffix(".dot")
            wp.capture_debug_dot_print(driver.graph, str(dot))
            graph = graph_summary(dot)
            dot.unlink()
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert not any("DtoH" in kind for kind in graph["memcpy_kinds"])
            if args.backend == "cg":
                assert not any("HtoD" in kind for kind in graph["memcpy_kinds"])
        prior_digest = hashlib.sha256()
        for array in (prior.indptr, prior.indices, prior.data):
            prior_digest.update(array.tobytes())
        report = dict(
            arguments=vars(args) | dict(out=str(args.out)),
            n=p.n,
            outer_capture="arpack"
            if driver is None
            else "adaptive"
            if driver.adaptive
            else "fixed",
            prior_rank_bound=args.columns if args.prior == "localized" else p.n,
            reference=ref["method"],
            guard=ref["guard"],
            assembly_seconds=assembly_seconds,
            setup_seconds=setup_seconds,
            capture_seconds=capture_seconds,
            reference_seconds=reference_seconds,
            warmup=warmup,
            rows=rows,
            graph=graph,
            matrix_sha256=matrix_hash(p),
            prior_sha256=prior_digest.hexdigest(),
            source_sha256=source_manifest(__file__),
            dependency=warp_dependencies(),
            accepted=all(r["accepted"] for r in [warmup] + rows),
        )
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if not report["accepted"]:
            raise SystemExit("Rejected implicit covariance solve; evidence retained")
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()
