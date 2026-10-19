"""Accuracy-gated normal-product POD comparison against existing implementations."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.compare import other_gpu_processes, run, timed
from benchmarks.applications.pod_compare import snapshots
from benchmarks.applications.problems import make_problem
from benchmarks.applications.runner import accuracy
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import NormalSVD
from warpack.normal_svd import _FullNormalCore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=["tall", "wide", "clustered", "wave", "application"], default="clustered"
    )
    p.add_argument("--size", type=int, default=100000)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--snapshots", type=int, default=48)
    p.add_argument("--full-basis-products", action="store_true")
    p.add_argument("--ncv", type=int, default=48)
    p.add_argument("--reference-ncv", type=int, default=40)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--compare", default="warp,cupy,scipy,spectra")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    methods = args.compare.split(",") if args.compare else []
    if set(methods) - {"warp", "cupy", "scipy", "spectra"} or args.repeats < 1:
        p.error("require valid comparison methods and positive repeats")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    competing = set(other_gpu_processes())
    problem = (
        make_problem("I", args.size)
        if args.case == "application"
        else snapshots(args.size, args.case, args.modes, columns=args.snapshots)
    )
    if problem.k != args.modes:
        p.error("the standard application fixture requires --modes 4")
    solver_type = NormalSVD
    if args.full_basis_products:
        from .full_normal_basis import FullBasisNormalSVD

        solver_type = FullBasisNormalSVD
    setup, solver = timed(
        lambda: solver_type(matrix(problem.a), problem.k, ncv=args.ncv, tol=1e-11)
    )
    capture, graph = timed(lambda: solver.capture(args.steps))
    dot = Path("build/performance-followup") / (destination.stem + ".dot")
    dot.parent.mkdir(parents=True, exist_ok=True)
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    dot.unlink()
    warmup, _ = timed(lambda: wp.capture_launch(graph))
    samples, checks = [], []
    for _ in range(args.repeats):
        competing.update(other_gpu_processes())
        seconds, _ = timed(lambda: wp.capture_launch(graph))
        competing.update(other_gpu_processes())
        check = accuracy(
            problem, solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T
        )
        check.update(
            status=int(solver.core.status.numpy()[0]),
            left_converged=int(solver.count_left.numpy()[0]),
            right_converged=int(solver.count_right.numpy()[0]),
            orthogonality_errors=solver.orthogonality_errors.numpy().tolist(),
            outer_iterations=int(solver.core.iterations.numpy()[0]),
            independent_recoveries=int(solver.core.recoveries.numpy()[0]),
        )
        check["accepted"] &= (
            check["status"] == 0
            and check["left_converged"] == check["right_converged"] == problem.k
        )
        samples.append(seconds)
        checks.append(check)
    matrix_hash = hashlib.sha256()
    for array in (problem.a.indptr, problem.a.indices, problem.a.data):
        matrix_hash.update(array.tobytes())
    report = dict(
        matrix_sha256=matrix_hash.hexdigest(),
        args=vars(args),
        matrix_shape=problem.a.shape,
        formulation="complete_coordinate_projection"
        if isinstance(solver.core, _FullNormalCore)
        else "matrix_free_krylov",
        gpu=wp.get_device().name,
        versions={
            n: importlib.metadata.version(n)
            for n in ("warp-lang", "numpy", "scipy", "cupy-cuda12x")
        },
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("warpack/normal_svd.py"),
                Path("warpack/sparse_utils.py"),
                Path(__file__).with_name("full_normal_basis.py"),
                Path("benchmarks/applications/pod_compare.py"),
                Path("warpack/solver.py"),
                Path("warpack/projected.py"),
                Path("warpack/kernels.py"),
                Path("warpack/svd.py"),
                Path("warpack/krylov_kernels.py"),
            )
        },
        protocol="Sequential methods on identical input. Synchronized warm replays restore starts. Assembly/reference/validation excluded. Normal method has its own explicit ncv; reference methods use reference_ncv. Pure Warp; normal equations square conditioning. Physical triplets and requested spaces must pass unchanged 1e-8 gates and normal method strict 1e-11 counts. "
        + (
            "Full-basis products explicitly form the small normal matrix by projecting onto the entire coordinate basis; no Krylov expansion."
            if isinstance(solver.core, _FullNormalCore)
            else "Matrix-free Krylov normal products; no formed Gram operator."
        ),
        normal=dict(
            setup_seconds=setup,
            capture_seconds=capture,
            warmup_seconds=warmup,
            ncv=solver.core.m,
            samples_seconds=samples,
            solve_seconds=float(np.median(samples)),
            validation=checks,
            accepted=all(c["accepted"] for c in checks),
            graph=audit,
            timing_uncontended=not competing,
        ),
        references=[],
    )
    solver.graph = solver.core.graph = None
    graph = solver = None
    gc.collect()
    for method in methods:
        row = run(
            "I", args.size, method, args.repeats, args.reference_ncv, args.steps, problem=problem
        )
        report["references"].append(row)
        print(method, row["outcome"], row.get("solve_seconds"), flush=True)
    competing.update(other_gpu_processes())
    report["competing_gpu_processes"] = sorted(competing)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("normal", report["normal"]["accepted"], report["normal"]["solve_seconds"], flush=True)
    if not report["normal"]["accepted"] or competing:
        raise SystemExit("Normal formulation accuracy or occupancy gate failed")


if __name__ == "__main__":
    main()
