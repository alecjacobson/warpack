"""Device-only CG experiments; host libraries are used for validation only."""

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear

from benchmarks.anvil import clocked, validate
from benchmarks.audit_profile import graph_summary
from benchmarks.cg_recycling import FreshKrylovSchur, WarmCGInverse
from warpack import CGInverse, CuDSSInverse, KrylovSchur, OrthogonalComplementOperator
from warpack.fem import assemble_rest, mass_normalized, read_mesh, rigid_body_basis


def batch_device_checks(batch_size):
    """Diagnostic only: retain upstream CG kernels, group its device loop body.

    This patches only this benchmark process. It does not change installed files
    or the production CGInverse implementation.
    """
    from warp._src.optim import linear as implementation

    original = implementation._run_capturable_loop
    signature = inspect.signature(original)

    def grouped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["check_every"] == 0 and bound.arguments["cycle_size"] == 1:
            cycle = bound.arguments["do_cycle"]

            def group():
                for _ in range(batch_size):
                    cycle()

            bound.arguments["do_cycle"] = group
            bound.arguments["cycle_size"] = batch_size
        return original(**bound.arguments)

    implementation._run_capturable_loop = grouped


def prepare_reference(h, mass, sa):
    """Create a validation reference, never used as an optimized starting guess."""
    Path("build").mkdir(exist_ok=True)
    if Path("build/anvil_reference.npz").exists() and Path("build/anvil_initial.npy").exists():
        return
    inverse = CuDSSInverse(mass_normalized(h, mass, shift=1), 1)
    solver = KrylovSchur(inverse, 26, ncv=64, which="LA", tol=1e-12)
    solver.initialize()
    np.save("build/anvil_initial.npy", solver.views[0].numpy().ravel())
    solver.solve(1)
    quality, (vectors, values) = validate(sa, solver.eigenvectors.numpy(), None)
    if not quality["accepted"]:
        raise RuntimeError("cuDSS validation reference failed the original-problem residual check")
    np.savez("build/anvil_reference.npz", vectors=vectors, values=values)
    inverse.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["linear", "eigen"], default="linear")
    p.add_argument("--shifts", type=float, nargs="+", default=[1, 100, 1000, 10000])
    p.add_argument("--preconditioners", nargs="+", default=["bj"])
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--ncv", type=int, default=64)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warm-start", default="zero")
    p.add_argument("--project-rigid", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--device-check-batch", type=int, default=1)
    p.add_argument("--out", default="build/cg_experiments.json")
    args = p.parse_args()
    if args.device_check_batch < 1:
        p.error("--device-check-batch must be positive")
    if args.device_check_batch > 1:
        batch_device_checks(args.device_check_batch)
    mesh = Path("anvil for export-ftetwild.mesh")
    x, t = read_mesh(mesh)
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    a.nnz_sync()
    n = a.shape[0]
    sa = sp.bsr_matrix(
        (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()), shape=(n, n)
    ).tocsr()
    prepare_reference(h, mass, sa)
    reference_file = np.load("build/anvil_reference.npz")
    reference = (reference_file["vectors"], reference_file["values"])
    initial = np.load("build/anvil_initial.npy")
    initial_device = wp.array(initial, dtype=wp.float64)
    projection_setup, rigid = clocked(
        lambda: rigid_body_basis(wp.array(x, dtype=wp.vec3d), mass) if args.project_rigid else None,
        gpu=True,
    )
    report = {
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "warp_version": wp.__version__,
        "linear_module_sha256": hashlib.sha256(
            Path(inspect.getfile(linear.cg)).read_bytes()
        ).hexdigest(),
        "arguments": vars(args),
        "process_id": os.getpid(),
        "initial_vector_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "rigid_basis_setup_seconds": projection_setup,
        "rows": [],
    }
    for shift in args.shifts:
        shifted = mass_normalized(h, mass, shift=shift)
        shifted.nnz_sync()
        for name in args.preconditioners:
            wp.synchronize()
            start = time.perf_counter()
            if name == "cudss":
                pre = None
            elif name == "bj":
                pre = linear.preconditioner(shifted, "block_jacobi_sequential")
            else:
                width = int(name.removeprefix("fsai").removesuffix("f"))
                pre = linear.FSAI(
                    shifted,
                    max_row_size=width,
                    max_step_size=3,
                    apply_lanes=8,
                    factor_dtype=wp.float32 if name.endswith("f") else wp.float64,
                )
            options = dict(tol=args.tol, maxiter=10000, preconditioner=pre)
            if name == "cudss":
                inv = CuDSSInverse(shifted, 1)
            else:
                inv = (
                    CGInverse(shifted, **options)
                    if args.warm_start == "zero"
                    else WarmCGInverse(shifted, args.warm_start, **options)
                )
            wp.synchronize()
            setup = time.perf_counter() - start
            print("Setup", shift, name, setup, flush=True)
            if args.mode == "linear":
                rhs = wp.array(initial.reshape(1, n), dtype=wp.float64)
                out = wp.empty_like(rhs)
                inv.apply(rhs, out)
                wp.synchronize()
                with wp.ScopedCapture() as capture:
                    if hasattr(inv, "reset_stats"):
                        inv.reset_stats()
                    if hasattr(inv, "reset_history"):
                        inv.reset_history()
                    inv.apply(rhs, out)
                graph = capture.graph
            else:
                operator = inv if rigid is None else OrthogonalComplementOperator(inv, rigid)
                solver = FreshKrylovSchur(
                    operator,
                    26 if rigid is None else 20,
                    ncv=args.ncv,
                    which="LA",
                    tol=1e-12,
                    initial_vector=initial_device,
                )
                capture_time, graph = clocked(
                    lambda: solver.capture(args.cycles, adaptive=False), gpu=True
                )
                print("Capture", capture_time, flush=True)
            warmup_seconds = None
            if args.repeats > 1:
                warmup_seconds, _ = clocked(lambda: wp.capture_launch(graph), gpu=True)
            rows = []
            begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            for trial in range(args.repeats):
                if hasattr(inv, "reset_stats"):
                    inv.reset_stats()
                wp.synchronize()
                window_start = time.time()
                wp.record_event(begin)
                start = time.perf_counter()
                wp.capture_launch(graph)
                enqueue = time.perf_counter() - start
                wp.record_event(end)
                wp.synchronize()
                elapsed = time.perf_counter() - start
                window_end = time.time()
                row = {
                    "seconds": elapsed,
                    "device_seconds": wp.get_event_elapsed_time(begin, end, False) / 1000,
                    "host_enqueue_seconds": enqueue,
                    "cg_statistics": inv.stats.numpy().tolist() if hasattr(inv, "stats") else None,
                    "measurement_unix_window": [window_start, window_end],
                }
                if args.mode == "linear":
                    answer = out.numpy()[0]
                    row["true_relative_residual"] = float(
                        np.linalg.norm(sa @ answer + shift * answer - initial)
                        / np.linalg.norm(initial)
                    )
                else:
                    vectors = solver.eigenvectors.numpy()
                    if rigid is not None:
                        vectors = np.concatenate([rigid.numpy(), vectors])
                    quality, _ = validate(sa, vectors, reference)
                    row.update(quality)
                rows.append(row)
                print(
                    "Trial",
                    shift,
                    name,
                    trial,
                    elapsed,
                    row["cg_statistics"],
                    row.get("max_original_residual", row.get("true_relative_residual")),
                    flush=True,
                )
            result = {
                "shift": shift,
                "preconditioner": name,
                "setup_seconds": setup,
                "solve_seconds": float(np.median([v["seconds"] for v in rows])),
                "warmup_seconds": warmup_seconds,
                "trials": rows,
            }
            if args.mode == "eigen":
                result["capture_seconds"] = capture_time
                result["accepted"] = all(v["accepted"] for v in rows)
            if args.profile:
                dot_path = str(Path("build") / Path(args.out).with_suffix(".dot").name)
                wp.capture_debug_dot_print(graph, dot_path, verbose=True)
                result["graph"] = graph_summary(dot_path)
                micro = {}
                rhs = wp.array(initial, dtype=wp.float64)
                output = wp.empty_like(rhs)
                for label, op in [
                    ("matrix", linear.aslinearoperator(shifted)),
                    ("preconditioner", pre),
                ]:
                    if op is None:
                        continue
                    op.matvec(rhs, output, output, 1.0, 0.0)
                    wp.synchronize()
                    with wp.ScopedCapture() as cap:
                        for _ in range(300):
                            op.matvec(rhs, output, output, 1.0, 0.0)
                    times = [
                        clocked(lambda: wp.capture_launch(cap.graph), gpu=True)[0] for _ in range(3)
                    ]
                    micro[label] = float(np.median(times) / 300)
                result["standalone_matvec_seconds"] = micro
            report["rows"].append(result)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            if hasattr(inv, "close"):
                inv.close()
            graph = inv = pre = None


if __name__ == "__main__":
    main()
