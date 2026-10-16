"""Computed starts for experimental block solves after a regional FEM stiffness change."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.applications.compare import other_gpu_processes, timed
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
    )


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
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not np.isfinite(args.change) or args.change <= 0 or args.repeats < 1:
        p.error("Require a finite positive stiffness change and repeats >= 1")
    before = other_gpu_processes()
    case = make_fem_case(args.case, args.side, args.modes)
    source = case.problem
    print("Prepared", args.case, "n=", source.n, flush=True)
    mass = SparseOperator(matrix(source.b))
    random = wp.array(
        np.random.default_rng(42).normal(size=(args.width, source.n)), dtype=wp.float64
    )
    source_setup, old_pre = timed(
        lambda: fsai(source.a - source.target * source.b, args.fsai_width)
    )
    bootstrap = BlockLOBPCG(
        SparseOperator(matrix(source.a)),
        source.k,
        args.width,
        mass=mass,
        preconditioner=UpstreamPreconditioner(old_pre),
        initial_vectors=random,
        tol=1e-10,
    )
    source_capture, source_graph = timed(lambda: bootstrap.capture(args.steps))
    source_seconds, _ = timed(lambda: wp.capture_launch(source_graph))
    source_quality = check(source, bootstrap)
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
    setup, current_pre = timed(lambda: fsai(changed - source.target * source.b, args.fsai_width))
    op = SparseOperator(matrix(changed))
    partial = wp.clone(random)
    wp.copy(partial[: source.k], previous[: source.k])
    report = dict(
        args=vars(args),
        n=source.n,
        reference=label,
        start_provenance="Previous modes/full block computed by this pure-Warp prototype; references only validate results.",
        source_setup_seconds=source_setup,
        source_capture_seconds=source_capture,
        source_solve_seconds=source_seconds,
        source_iterations=int(bootstrap.iterations.numpy()[0]),
        source_quality=source_quality,
        changed_preconditioner_setup_seconds=setup,
        rows=[],
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__), Path("benchmarks/experimental/block_lobpcg.py")]
        },
    )
    for name, start in [
        ("random", random),
        ("previous_modes", partial),
        ("previous_full_block", previous),
    ]:
        solver = BlockLOBPCG(
            op,
            source.k,
            args.width,
            mass=mass,
            preconditioner=UpstreamPreconditioner(current_pre),
            initial_vectors=start,
            tol=args.tol,
        )
        capture_seconds, graph = timed(lambda: solver.capture(args.steps))
        warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
        trials = []
        for _ in range(args.repeats):
            seconds, _ = timed(lambda: wp.capture_launch(graph))
            trials.append(
                dict(
                    seconds=seconds,
                    iterations=int(solver.iterations.numpy()[0]),
                    quality=check(target, solver),
                )
            )
        row = dict(
            start=name,
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            median_seconds=float(np.median([t["seconds"] for t in trials])),
            accepted=all(t["quality"]["accepted"] for t in trials),
            trials=trials,
        )
        report["rows"].append(row)
        print(name, row, flush=True)
    after = other_gpu_processes()
    report.update(
        other_gpu_processes_before=before,
        other_gpu_processes_after=after,
        timing_eligible=not (before or after),
        timing_note="Same changed pencil, preconditioner and stopping criteria for every start; no GPU speed claim if contention is observed. Source solve and setup costs are separate.",
    )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
