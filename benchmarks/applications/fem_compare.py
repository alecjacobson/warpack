"""Isolated elastic-ROM comparisons with complete requested-space acceptance."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import warp as wp
import warp.optim.linear as linear

from benchmarks.acceptance import add_diagnostic_argument, assess_comparison, require_accepted

from .compare import compare_baseline, run
from .fem_cases import make_fem_case
from .fem_projection import physical_problem, rigid_inverse_transform

NAMES = [
    "clamped_lumped",
    "clamped_consistent",
    "free",
    "disconnected",
    "heterogeneous",
    "repeated_bending",
]
METHODS = [
    "scipy",
    "spectra",
    "cupy",
    "cupy-eigsh",
    "cupy-batched",
    "warp",
    "warp-block",
    "warp-cg",
    "warp-cg-bj",
    "warp-cg-fsai",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(NAMES))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--side", type=int, default=2)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--ncv", type=int, default=48)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=-0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lobpcg-tol", type=float, default=1e-8)
    parser.add_argument("--cg-tol", type=float, default=1e-13)
    parser.add_argument("--cg-atol", type=float)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default="results/fem_comparisons.json")
    parser.add_argument("--baseline")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--project-rigid", action="store_true")
    add_diagnostic_argument(parser)
    args = parser.parse_args()
    names, methods = args.cases.split(","), args.methods.split(",")
    if set(names) - set(NAMES) or set(methods) - set(METHODS):
        parser.error("Unknown fixture or method")
    if args.project_rigid and (
        set(names) - {"free", "disconnected"}
        or set(methods) - {"warp", "warp-cg", "warp-cg-bj", "warp-cg-fsai"}
    ):
        parser.error("Rigid projection requires free/disconnected cases and a one-RHS Warp method")
    if not math.isfinite(args.cg_tol) or args.cg_tol <= 0:
        parser.error("CG tolerance must be finite and positive")
    if args.cg_atol is not None and (not math.isfinite(args.cg_atol) or args.cg_atol < 0):
        parser.error("CG absolute tolerance must be finite and nonnegative")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(names) != 1 or len(methods) != 1:
            parser.error("One fixture and method per worker")
        c = make_fem_case(names[0], args.side, args.modes, args.shift)
        original = c.problem
        p = physical_problem(c) if args.project_rigid else original
        row = run(
            "A",
            p.n,
            methods[0],
            args.repeats,
            args.ncv,
            args.steps,
            problem=p,
            lobpcg_tol=args.lobpcg_tol,
            cg_tol=args.cg_tol,
            cg_atol=args.cg_atol,
            warp_inverse_transform=rigid_inverse_transform(c) if args.project_rigid else None,
        )
        row.update(
            fixture=names[0],
            side=args.side,
            requested_nontrivial_modes=args.modes,
            nontrivial_modes=c.nontrivial_modes,
            known_rigid_modes=original.nullity,
            projected_rigid_modes=original.nullity if args.project_rigid else 0,
            reference=c.reference,
            reference_positive_guard=float(c.spectrum[original.k]),
            nullspace_policy="Retain all known rigid modes in the solve, then use the nontrivial block for ROM. No automatic weighted deflation API is implied.",
            boundary_conditions="Free" if original.nullity else "Clamped",
            material="E contrast 20, density contrast 3"
            if names[0] == "heterogeneous"
            else "Uniform",
            mass="Lumped" if names[0] == "clamped_lumped" else "Consistent",
            tetrahedra=None
            if c.tets is None
            else len(c.tets) * (2 if names[0] == "disconnected" else 1),
            worker_wall_limit_seconds=args.timeout,
        )
        if args.project_rigid:
            row["nullspace_policy"] = (
                "Analytic rigid basis from device geometry; normalize in M on device; P (K-sigma*M)^-1 P.T; return only the complete positive elastic block. No reference eigenvectors used as starts."
            )
        destination.write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        upstream_source_sha256={
            "linear": hashlib.sha256(
                Path(inspect.getfile(linear.preconditioner)).read_bytes()
            ).hexdigest(),
            "fsai": hashlib.sha256(Path(inspect.getfile(linear.FSAI)).read_bytes()).hexdigest()
            if hasattr(linear, "FSAI")
            else None,
        },
        protocol="Sequential isolated workers, every synchronized timed trial checked in the original pencil at 1e-8 for residual, spectrum, M-orthogonality, known rigid subspace and complete requested subspace. Full boundary clusters included. Reference assembly/factorization, setup/capture, output conversion and acceptance checks excluded from solve timing. cuDSS factors SPD; pure Warp CG profiles use upstream diagonal, block-Jacobi or FSAI preconditioning as recorded per row. CuPy is generalized LOBPCG+cuDSS (cupy-batched pads shrinking blocks to a single k-RHS solve); cupy-eigsh is a separate native path for positive diagonal mass with explicit scaling. Rejected/unsupported rows retained. Fixed outer budgets and stricter reported solver counts recorded separately.",
        cases=[],
    )
    folder = Path("build/fem-workers")
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        for method in methods:
            output = (
                folder
                / f"{name}-{args.side}-{args.modes}-{method}-cycles{args.steps}-shift{args.shift:g}-tol{args.lobpcg_tol:g}-project{int(args.project_rigid)}-cg{args.cg_tol:g}.json"
            )
            log = output.with_suffix(".log")
            command = [
                sys.executable,
                "-m",
                "benchmarks.applications.fem_compare",
                "--worker",
                "--cases",
                name,
                "--methods",
                method,
                "--side",
                str(args.side),
                "--modes",
                str(args.modes),
                "--ncv",
                str(args.ncv),
                "--steps",
                str(args.steps),
                "--shift",
                str(args.shift),
                "--repeats",
                str(args.repeats),
                "--lobpcg-tol",
                str(args.lobpcg_tol),
                "--out",
                str(output),
                "--timeout",
                str(args.timeout),
                "--cg-tol",
                str(args.cg_tol),
            ]
            if args.cg_atol is not None:
                command += ["--cg-atol", str(args.cg_atol)]
            if args.project_rigid:
                command.append("--project-rigid")
            output.unlink(missing_ok=True)
            try:
                with log.open("w") as f:
                    subprocess.run(command, stdout=f, stderr=f, check=True, timeout=args.timeout)
                row = json.loads(output.read_text())
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                row = dict(
                    fixture=name,
                    side=args.side,
                    requested_nontrivial_modes=args.modes,
                    method=method,
                    accepted=False,
                    outcome="timeout"
                    if isinstance(e, subprocess.TimeoutExpired)
                    else "worker_error",
                    log=str(log),
                    worker_wall_limit_seconds=args.timeout,
                )
            report["cases"].append(row)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(name, method, row["outcome"], row.get("solve_seconds"), flush=True)
    assess_comparison(report, report["cases"], args.diagnostic)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if args.baseline:
        old = json.loads(Path(args.baseline).read_text())
        if (old["gpu"], old["warp"]) != (report["gpu"], report["warp"]):
            raise ValueError("Baseline GPU/runtime mismatch")

        def key(r):
            return r["fixture"], r["side"], r["requested_nontrivial_modes"], r["method"]

        index = {key(r): r for r in old["cases"]}
        checks = []
        for row in report["cases"]:
            prior = index.get(key(row))
            if prior is None:
                continue
            if "n" not in row:
                checks.append(
                    dict(
                        fixture=row["fixture"],
                        method=row["method"],
                        comparable=False,
                        accuracy_regression=bool(prior.get("accepted")),
                    )
                )
            else:
                checks.extend(
                    dict(c, fixture=row["fixture"])
                    for c in compare_baseline([row], dict(cases=[prior]), 1.25)
                )
        report["regression_checks"] = checks
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if any(
            c.get("performance_regression") or c["accuracy_regression"] or c.get("graph_regression")
            for c in checks
        ):
            raise SystemExit(1)
        if not any(c["comparable"] for c in checks):
            raise SystemExit("No comparable uncontended timings")
    require_accepted(report)


if __name__ == "__main__":
    main()
