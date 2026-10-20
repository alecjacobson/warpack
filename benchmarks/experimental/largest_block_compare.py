"""Accuracy-gated largest-algebraic block profiles on the negative-FEM fixture."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import warp as wp
import warp.optim.linear as linear

from benchmarks.acceptance import add_diagnostic_argument, assess_comparison, require_accepted
from benchmarks.applications.compare import other_gpu_processes, run, timed
from benchmarks.applications.negative_foundation import make_negative_foundation
from benchmarks.applications.runner import accuracy
from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.largest_block import LargestBlock, upper_shifted_matrix
from benchmarks.upstream.adapters import matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", type=int, default=24)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--masses", default="lumped,consistent")
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-13)
    p.add_argument(
        "--preconditioner",
        choices=["none", "mass-diagonal", "upper-fsai", "upper-cg"],
        default="mass-diagonal",
    )
    p.add_argument("--fsai-width", type=int, default=32)
    p.add_argument("--inner-cg", type=int, default=8)
    p.add_argument("--inner-tol", type=float, default=0.01)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--references", action="store_true")
    p.add_argument("--reference-ncv", type=int, default=96)
    p.add_argument("--reference-steps", type=int, default=12)
    p.add_argument("--out", required=True)
    add_diagnostic_argument(p)
    args = p.parse_args()
    if set(args.masses.split(",")) - {"lumped", "consistent"} or min(args.steps, args.repeats) < 1:
        p.error("require valid masses and positive steps/repeats")
    report = dict(
        args=vars(args),
        gpu=wp.get_device().name,
        versions={
            n: importlib.metadata.version(n)
            for n in ["warp-lang", "numpy", "scipy", "cupy-cuda12x"]
        },
        cases=[],
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                *sorted(Path("warpack").glob("*.py")),
                Path(__file__),
                Path("benchmarks/experimental/largest_block.py"),
                Path("benchmarks/experimental/batched_cg.py"),
                Path("benchmarks/experimental/block_kernels.py"),
                Path("benchmarks/acceptance.py"),
                Path("benchmarks/applications/negative_foundation.py"),
            ]
        },
        protocol="Largest algebraic block solve on -K, signs restored on device. This fixture independently verifies that LA equals its LM target; no general LM claim. Seed42 block starts, native reference seed2026; full repeated requested spaces validated analytically. Warm synchronized replay excludes setup/capture/validation. Pure Warp; optional references use their declared backends. Both normalized outer counts and the fresh native absolute bound are required.",
    )
    competing = set(other_gpu_processes())
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for mass_name in args.masses.split(","):
        case = make_negative_foundation(args.side, args.modes, "LM", mass_name == "consistent")
        problem = case.problem
        if not case.largest_magnitude_is_largest_algebraic:
            p.error(
                "LA does not equal LM on this fixture; cannot use the sign-transform comparison"
            )
        digest = hashlib.sha256()
        for a in (problem.a, problem.b):
            for value in (a.indptr, a.indices, a.data):
                digest.update(value.tobytes())

        # Tensor-product Q1 stencil bound: each 1D eigenvalue is below
        # 12/h^2 (consistent) or 4/h^2 (lumped); material weights sum to 5.
        # This uses fixture inputs, not computed eigenpairs or reference starts.
        upper_shift = (60.0 if mass_name == "consistent" else 20.0) * (
            args.side + 1
        ) ** 2 - case.foundation

        def setup():
            a, b = matrix(problem.a), matrix(problem.b)
            pre = (
                linear.preconditioner(b, "diag") if args.preconditioner == "mass-diagonal" else None
            )
            if args.preconditioner.startswith("upper-"):
                shifted = upper_shifted_matrix(a, b, upper_shift)
                factor = linear.FSAI(
                    shifted,
                    max_row_size=args.fsai_width,
                    max_step_size=3,
                    apply_lanes=8,
                    factor_dtype=wp.float32,
                )
                pre = factor
                if args.preconditioner == "upper-cg":
                    from benchmarks.experimental.batched_cg import BatchedCGInverse

                    pre = BatchedCGInverse(
                        shifted,
                        args.width,
                        tol=args.inner_tol,
                        atol=0.0,
                        maxiter=args.inner_cg,
                        preconditioner=factor,
                        working_dtype=wp.float32,
                    )
            return LargestBlock(
                a,
                problem.k,
                mass=b,
                width=args.width,
                preconditioner=pre,
                tol=args.tol,
                gram_chunk=1024,
                symmetric_products=True,
                fused_updates=True,
                paired_rotation=True,
            )

        setup_seconds, solver = timed(setup)
        capture_seconds, graph = timed(lambda: solver.capture(args.steps))
        warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
        checks, samples = [], []
        for _ in range(args.repeats):
            competing.update(other_gpu_processes())
            seconds, _ = timed(lambda: wp.capture_launch(graph))
            competing.update(other_gpu_processes())
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            check = accuracy(problem, w, v)
            check["physical_accepted"] = check["accepted"]
            check.update(
                solver_status=int(solver.status.numpy()[0]),
                reported_converged=int(solver.converged.numpy()[0]),
                outer_iterations=int(solver.iterations.numpy()[0]),
                reported_max_residual=float(solver.residuals.numpy().max()),
            )
            check["solver_full_count"] = check["reported_converged"] == problem.k
            residual = problem.a @ v - (problem.b @ v) * w
            if mass_name == "lumped":
                absolute = float(np.linalg.norm(residual / np.sqrt(problem.b.diagonal())[:, None]))
                check["standard_form_frobenius_residual"] = absolute
            else:
                absolute = float(np.linalg.norm(residual, axis=0).max())
                check["original_max_absolute_residual"] = absolute
            check["native_stopping_passed"] = absolute <= 1e-8
            check["accepted"] &= (
                check["solver_status"] == 0
                and check["solver_full_count"]
                and check["native_stopping_passed"]
            )
            if hasattr(solver.preconditioner, "stats"):
                check["preconditioner_cg_stats"] = solver.preconditioner.stats.numpy().tolist()
            samples.append(seconds)
            checks.append(check)
        dot = Path("build") / (destination.stem + "-" + mass_name + ".dot")
        dot.parent.mkdir(exist_ok=True)
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        dot.unlink()
        row = dict(
            method="warp-largest-block",
            mass=mass_name,
            n=problem.n,
            nnz=problem.a.nnz,
            mass_nnz=problem.b.nnz,
            k=problem.k,
            width=args.width,
            projected_dimension=3 * args.width,
            preconditioner_upper_shift=upper_shift
            if args.preconditioner.startswith("upper-")
            else None,
            preconditioner_policy=f"Upstream FSAI max_row_size={args.fsai_width}, FP32 factor storage, optional intentionally truncated batched FP32 CG; bound derived from Q1 stencil inputs, not eigenpairs"
            if args.preconditioner.startswith("upper-")
            else args.preconditioner,
            matrix_sha256=digest.hexdigest(),
            setup_seconds=setup_seconds,
            setup_scope="Matrix uploads, shifted-matrix/factor construction, solver allocation and JIT where encountered; capture/warmup separate",
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            samples_seconds=samples,
            solve_seconds=float(np.median(samples)),
            validation=checks,
            accepted=all(c["accepted"] for c in checks),
            graph=audit,
            timing_uncontended=not competing,
        )
        report["cases"].append(row)
        print(mass_name, row["method"], row["accepted"], row["solve_seconds"], flush=True)
        if args.references:
            for method in (
                ("warp-diagonal", "cupy-eigsh") if mass_name == "lumped" else ("warp", "cupy")
            ):
                reference = run(
                    problem.api_class,
                    problem.n,
                    method,
                    repeats=args.repeats,
                    ncv=args.reference_ncv,
                    steps=args.reference_steps,
                    problem=problem,
                    eigsh_tol=1e-8,
                    lobpcg_tol=1e-8,
                    cupy_which="LM",
                )
                reference["mass"] = mass_name
                report["cases"].append(reference)
                print(
                    mass_name,
                    method,
                    reference["accepted"],
                    reference.get("solve_seconds"),
                    flush=True,
                )
        destination.write_text(json.dumps(report, indent=2) + "\n")
    report["competing_gpu_processes"] = sorted(competing)
    assess_comparison(report, report["cases"], args.diagnostic)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if competing:
        raise SystemExit("GPU contention; not timing evidence")
    require_accepted(report)


if __name__ == "__main__":
    main()
