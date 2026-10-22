"""Compare coarse starts and repeated coarse corrections, charging source solves."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.coarse_correction import CoarseCG, CoarseCorrection, FixedFSAI
from benchmarks.experimental.geometric_cg import GeometricCG
from benchmarks.experimental.geometric_preconditioner import build_hierarchy
from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner, BatchedSparseOperator

from .fem_hierarchy import (
    BeamTransfer,
    attach_reference,
    audit,
    beam_problem,
    matrix_hash,
    transfer_stencil,
)
from .fem_hierarchy_tuning import source_usable
from .fem_shapes import ShapeReplay, elapsed

PROFILES = (
    "cg",
    "start-cg",
    "fsai",
    "additive",
    "balanced",
    "balanced-start",
    "cg64",
    "additive-inner-cg",
    "balanced-inner-cg",
    "geometric",
    "geometric-inner-cg",
    "geometric-fp32-cg",
    "geometric-fp32-bucketed",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=int, default=5)
    parser.add_argument("--fine", type=int, default=15)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--mass", choices=("consistent", "lumped"), default="consistent")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--coarse-tol", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner-cg", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.coarse, args.fine, args.modes, args.repeats, args.steps, args.inner_cg) < 1
        or args.fine <= args.coarse
        or args.fine % args.coarse
        or args.width < args.modes
        or not math.isfinite(args.coarse_tol)
        or args.coarse_tol <= 0
    ):
        parser.error("positive sizes, fine>coarse, integer refinement and width>=modes required")
    if args.profile in ("cg", "start-cg") and args.inner_cg != 25:
        parser.error("the existing bucketed CG baseline uses exactly 25 inner iterations")
    uses_geometry = args.profile.startswith("geometric")
    uses_coarse = args.profile not in ("cg", "cg64", "fsai") and not uses_geometry
    uses_correction = args.profile in (
        "additive",
        "balanced",
        "balanced-start",
        "additive-inner-cg",
        "balanced-inner-cg",
    )
    uses_start = args.profile in ("start-cg", "balanced-start")
    setup, problems = {}, {}
    for name, side, count in [("fine", args.fine, args.modes)] + (
        [("coarse", args.coarse, args.width)] if uses_coarse else []
    ):
        begin = time.perf_counter()
        problems[name] = beam_problem(side, count, args.mass == "consistent")
        setup[name + "_assembly_seconds"] = time.perf_counter() - begin
    begin = time.perf_counter()
    for p in problems.values():
        attach_reference(p)
    reference_seconds = time.perf_counter() - begin
    device = wp.get_device("cuda:0")
    memory_before = device.free_memory
    coarse, transfer, correction, hierarchy = None, None, None, None
    if uses_coarse:
        begin = time.perf_counter()
        coarse = ShapeReplay(
            [(0, problems["coarse"], 0)],
            args.width + args.width - args.modes,
            refit=False,
            tol=args.coarse_tol,
        )
        wp.synchronize()
        setup["coarse_setup_capture_seconds"] = time.perf_counter() - begin
        setup["coarse_initial_solve_seconds"] = elapsed(coarse.solve)
        basis = wp.empty((args.width, problems["fine"].n), dtype=wp.float64, device=coarse.a.device)
        begin = time.perf_counter()
        transfer = BeamTransfer(args.coarse, args.fine, coarse.solver.eigenvectors, basis)
        wp.synchronize()
        setup["transfer_setup_seconds"] = time.perf_counter() - begin

    def factory(shifted, factor, width):
        nonlocal correction, hierarchy
        if uses_geometry:
            transfers = []
            for fine_side, coarse_side in ((args.fine, args.coarse), (args.coarse, 1)):
                if fine_side == coarse_side:
                    continue
                ids, weights = transfer_stencil(coarse_side, fine_side)
                n = ids.shape[0]
                ids, weights = ids.ravel(), weights.ravel()
                valid = (ids >= 0) & (weights != 0)
                host = sp.csr_matrix(
                    (weights[valid], (np.repeat(np.arange(n), 4)[valid], ids[valid])),
                    shape=(n, 9 * coarse_side * (coarse_side + 1) ** 2),
                )
                transfers.append(matrix(host))
            hierarchy = build_hierarchy(
                shifted,
                transfers,
                width,
                fine_factor=factor,
                dtype=wp.float32 if "fp32" in args.profile else wp.float64,
            )
            if "fp32" in args.profile:
                return GeometricCG(shifted, hierarchy, maxiter=args.inner_cg)
            if args.profile.endswith("inner-cg"):
                return CoarseCG(shifted, hierarchy, width, maxiter=args.inner_cg)
            return hierarchy
        if args.profile == "cg64":
            return BatchedCGPreconditioner(
                shifted,
                width,
                maxiter=args.inner_cg,
                tol=0.01,
                preconditioner=factor,
                working_dtype=wp.float64,
                matrix_layout="columns",
                fsai_layout="columns",
            )
        smoother = FixedFSAI(factor, width)
        if not uses_correction:
            return smoother
        correction = CoarseCorrection(
            BatchedSparseOperator(shifted),
            basis,
            smoother,
            width,
            balanced=not args.profile.startswith("additive"),
        )
        if args.profile.endswith("inner-cg"):
            return CoarseCG(shifted, correction, width, maxiter=args.inner_cg)
        return correction

    begin = time.perf_counter()
    fine = ShapeReplay(
        [(0, problems["fine"], 0)],
        args.width,
        refit=False,
        steps=None,
        start_rows=args.width,
        preconditioner_factory=factory if args.profile not in ("cg", "start-cg") else None,
        preconditioner_bucket_size=4 if args.profile == "geometric-fp32-bucketed" else None,
    )
    wp.synchronize()
    setup["fine_setup_seconds"] = time.perf_counter() - begin

    def initialize():
        if transfer:
            transfer.apply()
        if correction:
            correction.update()
        if uses_start:
            wp.copy(fine.starts, basis)

    begin = time.perf_counter()
    fine.graph = fine.solver.capture(args.steps, initializer=initialize)
    wp.synchronize()
    setup["fine_capture_seconds"] = time.perf_counter() - begin
    memory_after_capture = device.free_memory
    rows = []
    for trial in range(args.repeats + 1):
        wp.copy(fine.starts, fine.random)
        source_seconds = elapsed(coarse.solve) if coarse else 0.0
        fine_seconds = elapsed(lambda: wp.capture_launch(fine.graph))
        quality = fine.quality(problems["fine"])
        source_quality = coarse.quality(problems["coarse"]) if coarse else None
        status = int((correction or hierarchy).status.numpy()[0]) if correction or hierarchy else 0
        row = dict(
            source_seconds=source_seconds,
            fine_seconds=fine_seconds,
            total_seconds=source_seconds + fine_seconds,
            quality=quality,
            source_quality=source_quality,
            correction_status=status,
            cg_stats=fine.cg_statistics(),
            accepted=bool(
                quality["accepted"]
                and status == 0
                and (
                    source_quality is None
                    or source_usable(source_quality, args.width, args.coarse_tol)
                )
            ),
        )
        print(trial, row, flush=True)
        if trial:
            rows.append(row)
        else:
            warmup = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    graphs = {"fine": audit(fine.graph, args.out.with_suffix(".fine.dot"))}
    if coarse:
        graphs["coarse"] = audit(coarse.graph, args.out.with_suffix(".coarse.dot"))
    files = [
        Path(__file__),
        Path("benchmarks/experimental/coarse_correction.py"),
        Path("benchmarks/experimental/geometric_preconditioner.py"),
        Path("benchmarks/experimental/geometric_cg.py"),
        Path("benchmarks/applications/fem_shapes.py"),
        Path("benchmarks/applications/fem_hierarchy.py"),
        Path("benchmarks/applications/fem_hierarchy_tuning.py"),
        Path("benchmarks/applications/fem_cases.py"),
        Path("benchmarks/applications/problems.py"),
        *sorted(Path("warpack").glob("*.py")),
    ]
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        setup=setup,
        setup_total_seconds=sum(setup.values()),
        setup_capture_device_free_memory_decrease_bytes=memory_before - memory_after_capture,
        memory_scope="Device/context free-memory decrease after setup and capture, not peak or live factor bytes; interpret only without competing GPU processes.",
        reference_seconds=reference_seconds,
        source_tolerance=args.coarse_tol if coarse else None,
        source_count=args.width if coarse else None,
        n=problems["fine"].n,
        warmup=warmup,
        rows=rows,
        graphs=graphs,
        dependency=warp_dependencies(type(fine.factor)),
        environment=dict(
            gpu=device.name,
            platform=platform.platform(),
            driver=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
        ),
        matrix_sha256={k: matrix_hash(p) for k, p in problems.items()},
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        accepted=all(r["accepted"] for r in [warmup] + rows),
        scope=(
            "Static geometric hierarchy, pure-Warp sparse Galerkin products and FSAI, "
            "with a 36-DOF Warp Cholesky bottom solve. Transfer assembly, all factors "
            "and capture charged to setup; no coarse eigenbasis/source solve. "
            "Independent references excluded. No default policy change."
            if uses_geometry
            else "Warp-computed full-width coarse basis, pure-Warp fixed-linear FSAI and Galerkin correction. Setup includes assembly, source warmup, workspaces and capture; each warm workflow charges coarse solve, transfer/correction refresh and fine solve. Independent references excluded. No multiscale Lanczos recurrence or default policy change."
        ),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
