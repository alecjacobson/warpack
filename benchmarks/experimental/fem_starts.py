"""Computed starts for experimental block solves after a regional FEM stiffness change."""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.applications.compare import other_gpu_processes, run, timed
from benchmarks.applications.fem_cases import elastic_matrices, make_fem_case, reference
from benchmarks.applications.runner import accuracy
from benchmarks.experimental.block_lobpcg import BlockLOBPCG, UpstreamPreconditioner
from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator


def fsai(host, width):
    blocks = host.tobsr(blocksize=(3, 3))
    device = ws.bsr_from_triplets(
        blocks.shape[0] // 3,
        blocks.shape[1] // 3,
        wp.array(
            np.repeat(np.arange(blocks.shape[0] // 3), np.diff(blocks.indptr)), dtype=wp.int32
        ),
        wp.array(blocks.indices, dtype=wp.int32),
        wp.array(blocks.data, dtype=wp.mat33d),
    )
    device.nnz_sync()
    return linear.FSAI(
        device, max_row_size=width, max_step_size=3, apply_lanes=8, factor_dtype=wp.float32
    ), device


def preconditioner(host, args):
    factor, device = fsai(host, args.fsai_width)
    if args.inner_cg:
        from benchmarks.experimental.batched_cg import BatchedCGInverse

        inverse = BatchedCGInverse(
            device,
            args.width,
            tol=args.inner_tol,
            maxiter=args.inner_cg,
            preconditioner=factor,
            working_dtype=wp.float32 if args.inner_fp32 else wp.float64,
        )
    else:
        inverse = UpstreamPreconditioner(factor)
    return factor, inverse


def with_active_preconditioner(solver, args):
    if args.active_cg_bucket_size:
        from benchmarks.experimental.active_cg import DeviceBucketedCG

        solver.preconditioner = DeviceBucketedCG(
            solver.preconditioner, solver.errors, solver.tol, args.active_cg_bucket_size
        )
    return solver


def inner_quality(solver):
    pre = solver.preconditioner
    result = {"preconditioner_cg_stats": pre.stats.numpy().tolist()}
    if hasattr(pre, "usage"):
        result["preconditioner_cg_bucket_widths"] = pre.widths.numpy().tolist()
        result["preconditioner_cg_bucket_usage"] = pre.usage.numpy().tolist()
    return result


def check(problem, solver):
    q = accuracy(problem, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
    q.update(status=int(solver.status.numpy()[0]), converged=int(solver.converged.numpy()[0]))
    q["accepted"] &= q["status"] == 0 and q["converged"] == problem.k
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=["clamped_consistent", "clamped_lumped"], default="clamped_consistent"
    )
    p.add_argument("--side", type=int, default=10)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fsai-width", type=int, default=32)
    p.add_argument("--change", type=float, default=0.02)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inner-cg", type=int, default=0)
    p.add_argument("--active-cg-bucket-size", type=int, default=0)
    p.add_argument("--inner-fp32", action="store_true")
    p.add_argument("--inner-tol", type=float, default=0.01)
    p.add_argument("--batched-products", action="store_true")
    p.add_argument("--gram-chunk", type=int, default=256)
    p.add_argument("--symmetric-products", action="store_true")
    p.add_argument("--fused-updates", action="store_true")
    p.add_argument("--paired-rotation", action="store_true")
    p.add_argument("--reuse-x-image", action="store_true")
    p.add_argument(
        "--initial-perturbation",
        type=float,
        default=0.0,
        help="Add a separate Warp full-warm-block trial with this relative exploration amplitude",
    )
    p.add_argument("--cupy", action="store_true")
    p.add_argument("--cupy-width", type=int)
    p.add_argument("--cupy-tol", type=float)
    p.add_argument("--cupy-padding", action="store_true", help="reproduce padded-CG diagnostics")
    p.add_argument(
        "--baselines",
        action="store_true",
        help="Compare changed pencil with production CG and cuDSS",
    )
    p.add_argument("--baseline-ncv", type=int, default=64)
    p.add_argument("--baseline-steps", type=int, default=2)
    p.add_argument("--baseline-cg-tol", type=float, default=1e-15)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    solver_type = BlockLOBPCG
    if args.reuse_x_image:
        from benchmarks.experimental.cached_x_image import CachedXBlockLOBPCG

        solver_type = CachedXBlockLOBPCG
    if not np.isfinite(args.change) or args.change <= 0 or args.repeats < 1:
        p.error("Require a finite positive stiffness change and repeats >= 1")
    if args.inner_cg < 0 or (args.inner_fp32 and not args.inner_cg):
        p.error("Require nonnegative inner-cg and inner-cg for inner-fp32")
    if args.cupy_tol is not None and (not np.isfinite(args.cupy_tol) or args.cupy_tol <= 0):
        p.error("Require finite positive cupy-tol")
    if not np.isfinite(args.initial_perturbation) or args.initial_perturbation < 0:
        p.error("Require finite nonnegative initial-perturbation")
    if args.active_cg_bucket_size < 0 or (args.active_cg_bucket_size and not args.inner_cg):
        p.error("active-cg-bucket-size requires a positive size and inner-cg")
    before = other_gpu_processes()
    case = make_fem_case(args.case, args.side, args.modes)
    source = case.problem
    if args.cupy_width is not None and not source.k <= args.cupy_width <= args.width:
        p.error("Require k <= cupy-width <= width")
    operator = SparseOperator
    if args.batched_products:
        from benchmarks.experimental.block_kernels import BatchedSparseOperator

        operator = BatchedSparseOperator
    print("Prepared", args.case, "n=", source.n, flush=True)
    mass_setup, mass = timed(lambda: operator(matrix(source.b)))
    random = wp.array(
        np.random.default_rng(42).normal(size=(args.width, source.n)), dtype=wp.float64
    )
    source_setup, (_, old_pre) = timed(
        lambda: preconditioner(source.a - source.target * source.b, args)
    )
    source_solver_setup, bootstrap = timed(
        lambda: with_active_preconditioner(
            solver_type(
                operator(matrix(source.a)),
                source.k,
                args.width,
                mass=mass,
                preconditioner=old_pre,
                initial_vectors=random,
                tol=1e-10,
                gram_chunk=args.gram_chunk,
                symmetric_products=args.symmetric_products,
                fused_updates=args.fused_updates,
                paired_rotation=args.paired_rotation,
            ),
            args,
        )
    )
    source_capture, source_graph = timed(lambda: bootstrap.capture(args.steps))
    source_seconds, _ = timed(lambda: wp.capture_launch(source_graph))
    source_quality = check(source, bootstrap)
    if args.inner_cg:
        source_quality.update(inner_quality(bootstrap))
    if not source_quality["accepted"]:
        raise RuntimeError(f"Source solve failed independent validation: {source_quality}")
    print("Source accepted", int(bootstrap.iterations.numpy()[0]), "iterations", flush=True)
    previous = wp.clone(bootstrap.x)
    selected = case.vertices[case.tets].mean(axis=1)[:, 0] > 1.5
    regional, _ = elastic_matrices(case.vertices, case.tets[selected])
    regional = regional[case.dofs, :][:, case.dofs]
    changed = source.a + args.change * regional
    values, vectors, label = reference(changed, source.b, source.k + 8)
    target = replace(
        source, a=changed, expected=values[: source.k], requested_basis=vectors[:, : source.k]
    )
    setup, (current_factor, current_pre) = timed(
        lambda: preconditioner(changed - source.target * source.b, args)
    )
    operator_setup, op = timed(lambda: operator(matrix(changed)))
    partial = wp.clone(random)
    wp.copy(partial[: source.k], previous[: source.k])
    report = dict(
        args=vars(args),
        n=source.n,
        nnz=target.a.nnz,
        mass_nnz=target.b.nnz,
        shift=float(source.target),
        seed=42,
        gpu=wp.get_device().name,
        versions={
            name: importlib.metadata.version(name)
            for name in ["warp-lang", "numpy", "scipy", "cupy-cuda12x"]
        },
        reference=label,
        start_provenance="Previous modes/full block computed by this pure-Warp prototype; references only validate results.",
        source_setup_seconds=source_setup,
        source_solver_setup_seconds=source_solver_setup,
        shared_mass_operator_setup_seconds=mass_setup,
        source_capture_seconds=source_capture,
        source_solve_seconds=source_seconds,
        source_iterations=int(bootstrap.iterations.numpy()[0]),
        source_quality=source_quality,
        changed_preconditioner_setup_seconds=setup,
        changed_operator_setup_seconds=operator_setup,
        setup_scope="Preconditioner setup includes upstream FSAI and optional batched CG state/copy. Optional active-bucket states are charged to solver setup. Solver/operator workspaces, capture, and source solve are separate. Host FEM assembly/reference and starting-block generation/upload are excluded. JIT is charged where encountered.",
        cg_iteration_accounting="Scheduled vector iterations including masked RHSs and within-bucket padding when enabled; CuPy padding also includes zero RHSs. These are not sums of individual first-convergence counts.",
        rows=[],
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path(__file__),
                Path("benchmarks/experimental/block_lobpcg.py"),
                Path("warpack/block.py"),
                Path("warpack/block_products.py"),
                Path("benchmarks/experimental/block_kernels.py"),
                Path("benchmarks/experimental/block_updates.py"),
                Path("benchmarks/experimental/batched_cg.py"),
                Path("benchmarks/experimental/active_cg.py"),
                Path("benchmarks/experimental/cached_x_image.py"),
                Path("benchmarks/experimental/compare_block.py"),
            ]
        },
    )
    starts = [
        ("random", random, 0.0),
        ("previous_modes", partial, 0.0),
        ("previous_full_block", previous, 0.0),
    ]
    if args.initial_perturbation:
        starts.append(("previous_full_block_perturbed", previous, args.initial_perturbation))
    for name, start, perturbation in starts:
        solver_setup, solver = timed(
            lambda: with_active_preconditioner(
                solver_type(
                    op,
                    source.k,
                    args.width,
                    mass=mass,
                    preconditioner=current_pre,
                    initial_vectors=start,
                    initial_perturbation=perturbation,
                    tol=args.tol,
                    gram_chunk=args.gram_chunk,
                    symmetric_products=args.symmetric_products,
                    fused_updates=args.fused_updates,
                    paired_rotation=args.paired_rotation,
                ),
                args,
            )
        )
        capture_seconds, graph = timed(lambda: solver.capture(args.steps))
        warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
        trials = []
        for _ in range(args.repeats):
            seconds, _ = timed(lambda: wp.capture_launch(graph))
            quality = check(target, solver)
            if args.inner_cg:
                quality.update(inner_quality(solver))
            trials.append(
                dict(
                    seconds=seconds,
                    iterations=int(solver.iterations.numpy()[0]),
                    quality=quality,
                )
            )
        row = dict(
            start=name,
            initial_perturbation=perturbation,
            solver_setup_seconds=solver_setup,
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            median_seconds=float(np.median([t["seconds"] for t in trials])),
            accepted=all(t["quality"]["accepted"] for t in trials),
            trials=trials,
        )
        report["rows"].append(row)
        print(name, row, flush=True)
        if args.cupy and not perturbation:
            from benchmarks.experimental.compare_block import cupy_trials

            cupy_rows, metadata = cupy_trials(
                target.a,
                target.b,
                current_factor,
                start.numpy().T,
                None,
                args,
                source.k,
                lambda w, v: accuracy(target, w, v),
                cg=current_pre if args.inner_cg else None,
            )
            row["cupy_trials"], row["cupy_adapter"] = cupy_rows, metadata
    if args.baselines:
        report["production_baselines"] = [
            run(
                target.api_class,
                target.n,
                method,
                repeats=args.repeats,
                ncv=args.baseline_ncv,
                steps=args.baseline_steps,
                problem=target,
                cg_tol=args.baseline_cg_tol,
            )
            for method in ("warp", "warp-cg-fsai")
        ]
    after = other_gpu_processes()
    report.update(
        other_gpu_processes_before=before,
        other_gpu_processes_after=after,
        timing_eligible=not (before or after),
        timing_note="Optional perturbed full-block trial measures the explicit Warp exploration policy; CuPy comparisons use only identical unperturbed starts. Same changed pencil, preconditioner and stopping criteria for every Warp start; no GPU speed claim if contention is observed. Source solve and setup costs are separate.",
    )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
