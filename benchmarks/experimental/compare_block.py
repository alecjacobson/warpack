"""Accuracy-gated experimental block-method comparisons; no production CuPy use."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.anvil import validate
from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.applications.fem_cases import make_fem_case
from benchmarks.applications.hodge import make_hodge_case
from benchmarks.applications.runner import accuracy
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.block_lobpcg import BlockLOBPCG, UpstreamPreconditioner
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import matrix
from warpack import CGInverse, SparseOperator

FEM_CASES = (
    "clamped_consistent",
    "clamped_lumped",
    "free",
    "disconnected",
    "heterogeneous",
    "repeated_bending",
)


def memory_snapshot():
    device = wp.get_device()
    snapshot = {"device_free_bytes": device.free_memory}
    for label, name in (
        ("warp_pool_used_bytes", "get_mempool_used_mem_current"),
        ("warp_pool_high_bytes", "get_mempool_used_mem_high"),
    ):
        query = getattr(wp, name, None)
        snapshot[label] = (
            query(device) if query is not None and device.is_mempool_supported else None
        )
    return snapshot


def prepare(args, *, metadata=None):
    if args.case == "anvil":
        from benchmarks import anvil_inputs

        arrays, manifest = anvil_inputs.load()
        reference, reference_identity = anvil_inputs.reference()
        a, shifted = anvil_inputs.shifted_device(arrays, args.shift)
        sa = anvil_inputs.host_matrix(arrays)
        constraints = wp.array(arrays["rigid"], dtype=wp.float64, device=a.device)
        if metadata is not None:
            metadata.update(anvil_fixture=manifest, reference=reference_identity)

        def check(w, v):
            quality = validate(
                sa,
                np.concatenate([constraints.numpy(), v.T]),
                reference,
            )[0]
            # The shared Anvil validator recomputes Rayleigh quotients. Also
            # check the eigenvalues actually returned by the tested solver.
            av = sa @ v
            residual = np.linalg.norm(av - v * w, axis=0) / np.maximum(
                1, np.linalg.norm(av, axis=0) + abs(w) * np.linalg.norm(v, axis=0)
            )
            quality.update(
                returned_eigenvalue_error=float(np.max(abs(w / reference[1][6:] - 1))),
                returned_pair_residual=float(residual.max()),
            )
            quality["accepted"] &= (
                quality["returned_eigenvalue_error"] < 1e-8
                and quality["returned_pair_residual"] < 1e-8
            )
            return quality

        return a, None, shifted, constraints, sa, None, 20, check
    if args.case.startswith("hodge_"):
        c = make_hodge_case(args.case.removeprefix("hodge_"), args.side)
        p = c.problem
        identity_mass = (p.b - sp.eye(p.n, format="csr")).nnz == 0
        return (
            matrix(p.a),
            None if identity_mass else matrix(p.b),
            matrix(p.a + args.shift * p.b),
            None,
            p.a,
            None if identity_mass else p.b,
            p.k,
            lambda w, v: accuracy(p, w, v),
        )
    c = make_fem_case(args.case, args.side, args.modes)
    p = c.problem
    shifted_host = p.a + args.shift * p.b
    # Tet elasticity groups xyz DOFs into 3x3 blocks. The bending fixture has
    # displacement/rotation pairs; use scalar CSR instead of regrouping them.
    if p.block_size == 3:
        blocks = shifted_host.tobsr(blocksize=(3, 3))
        shifted = ws.bsr_from_triplets(
            p.n // 3,
            p.n // 3,
            wp.array(np.repeat(np.arange(p.n // 3), np.diff(blocks.indptr)), dtype=wp.int32),
            wp.array(blocks.indices, dtype=wp.int32),
            wp.array(blocks.data, dtype=wp.mat33d),
        )
    else:
        shifted = matrix(shifted_host)
    if metadata is not None:
        metadata.update(
            reference=c.reference,
            requested_nontrivial_modes=args.modes,
            actual_nontrivial_modes=c.nontrivial_modes,
            retained_rigid_modes=p.nullity,
            boundary_cluster_complete=True,
            first_excluded_eigenvalue=float(c.spectrum[p.k]),
            preconditioner_block_size=3 if p.block_size == 3 else 1,
            notes=p.notes,
        )
    return (matrix(p.a), matrix(p.b), shifted, None, p.a, p.b, p.k, lambda w, v: accuracy(p, w, v))


def finalize_acceptance(report):
    """Require outer accuracy/counts and each reference's declared stopping rule.

    Truncated CG here is a preconditioner, not an exact inverse. Its diagnostic
    unfinished RHS count is therefore deliberately not an acceptance failure.
    """
    for row in report["rows"]:
        row["physical_and_outer_accepted"] = bool(row["quality"]["accepted"])
        row["native_stopping_accepted"] = bool(row.get("full_block_absolute_tolerance_met", True))
        row["accepted"] = row["physical_and_outer_accepted"] and row["native_stopping_accepted"]
    report["accepted"] = bool(report["rows"]) and all(row["accepted"] for row in report["rows"])
    report["accepted"] &= report.get("post_profile_quality", {}).get("accepted", True)
    return report["accepted"]


def collect_reference(report, run):
    """Keep completed trials if the optional reference fails during setup/solve."""
    try:
        report["cupy_adapter"] = run(report["rows"])
    except Exception as error:
        report["rows"].append(
            dict(
                method="CuPy LOBPCG reference failure",
                seconds=None,
                error_type=type(error).__name__,
                error=str(error),
                quality=dict(accepted=False),
                full_block_absolute_tolerance_met=False,
            )
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=[
            "anvil",
            *FEM_CASES,
            "hodge_large_disconnected",
            "hodge_torus",
            "hodge_disconnected",
            "hodge_weak_positive",
        ],
        default="clamped_consistent",
    )
    p.add_argument("--side", type=int, default=6)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--shift", type=float, default=0.01)
    p.add_argument("--fsai-width", type=int, default=64)
    p.add_argument("--fsai-layout", choices=["rows", "columns"], default="rows")
    p.add_argument("--matrix-layout", choices=["rows", "columns"], default="rows")
    p.add_argument("--preconditioner", choices=["fsai", "bj"], default="fsai")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--cupy", action="store_true")
    p.add_argument("--cupy-width", type=int, help="CuPy starting block width; default matches Warp")
    p.add_argument("--cupy-tol", type=float, help="explicit CuPy absolute residual tolerance")
    p.add_argument(
        "--cupy-padding", action="store_true", help="reproduce biased padded-CG diagnostics"
    )
    p.add_argument(
        "--cupy-partitioned",
        action="store_true",
        help="exact RHS coverage in power-of-two CG chunks with linear workspace storage",
    )
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--public-api", action="store_true", help="Use only installed solver constructors"
    )
    p.add_argument(
        "--keep-graph", action="store_true", help="Retain the potentially large DOT dump"
    )
    p.add_argument("--gram-chunk", type=int, default=256)
    p.add_argument("--symmetric-products", action="store_true")
    p.add_argument("--fused-updates", action="store_true")
    p.add_argument("--paired-rotation", action="store_true")
    p.add_argument("--reuse-x-image", action="store_true")
    p.add_argument("--active-search-bucket-size", type=int, default=0)
    p.add_argument("--active-search-keep-history", action="store_true")
    p.add_argument("--native-products", action="store_true")
    p.add_argument("--batched-products", action="store_true")
    p.add_argument("--inner-cg", type=int, default=0)
    p.add_argument("--batched-cg", action="store_true")
    p.add_argument(
        "--active-cg-bucket-size",
        type=int,
        default=0,
        help="Experimental device-only residual packing; zero disables it",
    )
    p.add_argument("--fixed-outer", action="store_true")
    p.add_argument("--inner-fp32", action="store_true")
    p.add_argument("--inner-tol", type=float, default=0.01)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cupy_partitioned and (
        args.cupy_padding or not args.cupy or not args.batched_cg or not args.inner_cg
    ):
        p.error("--cupy-partitioned needs --cupy --batched-cg and excludes --cupy-padding")
    if args.active_search_keep_history and not args.active_search_bucket_size:
        p.error("--active-search-keep-history requires active search")
    if args.active_search_bucket_size < 0:
        p.error("active-search-bucket-size must be nonnegative")
    if args.active_search_bucket_size and (not args.reuse_x_image or args.profile):
        p.error("active search requires --reuse-x-image and does not support --profile")
    if args.cupy and args.inner_cg and not args.batched_cg:
        p.error("CuPy iterative comparison requires --batched-cg")
    if args.inner_cg < 0 or args.repeats < 1:
        p.error("require inner-cg >= 0 and repeats >= 1")
    if args.cupy_tol is not None and (not np.isfinite(args.cupy_tol) or args.cupy_tol <= 0):
        p.error("cupy-tol must be finite and positive")
    if args.batched_cg and not args.inner_cg:
        p.error("--batched-cg requires --inner-cg")
    if args.inner_fp32 and not args.batched_cg:
        p.error("--inner-fp32 requires --batched-cg")
    if args.active_cg_bucket_size < 0 or (args.active_cg_bucket_size and not args.batched_cg):
        p.error("active-cg-bucket-size requires a positive size and batched-cg")
    if args.fsai_layout == "columns" and (not args.batched_cg or args.preconditioner != "fsai"):
        p.error("column FSAI layout requires batched CG with FSAI")
    if args.matrix_layout == "columns" and not args.batched_cg:
        p.error("column matrix layout requires batched CG")
    before = other_gpu_processes()
    fixture = {}
    a, mass, shifted, constraints, sa, sb, k, check = prepare(args, metadata=fixture)
    if args.width < k:
        p.error(f"width must be at least {k}, including rigid modes and complete boundary clusters")
    if 3 * args.width > a.shape[0]:
        p.error("block iteration requires 3*width <= n")
    if args.cupy_width is not None and not k <= args.cupy_width <= args.width:
        p.error("require k <= cupy-width <= width")
    print("Prepared", args.case, "n=", a.shape[0], flush=True)
    shifted.nnz_sync()
    memory_before = memory_snapshot()
    memory_before_setup = memory_before["device_free_bytes"]
    setup_seconds, pre = timed(
        lambda: (
            linear.FSAI(
                shifted,
                max_row_size=args.fsai_width,
                max_step_size=3,
                factor_dtype=wp.float32,
            )
            if args.preconditioner == "fsai"
            else linear.preconditioner(shifted, ptype="block_jacobi_sequential")
        )
    )
    memory_after_preconditioner = memory_snapshot()
    print("Preconditioner ready", setup_seconds, flush=True)
    initial = np.random.default_rng(42).normal(size=(a.shape[0], args.width))
    input_identity = None
    if args.case == "anvil":
        from benchmarks import anvil_inputs

        input_identity = anvil_inputs.inputs(
            fixture["anvil_fixture"],
            fixture["reference"],
            initial,
            anvil_inputs.host_device_matrix(shifted),
            constraints=constraints.numpy(),
        )

    def create_solver():
        solver_type = BlockLOBPCG
        solver_options = {}
        if args.reuse_x_image and not args.public_api:
            from benchmarks.experimental.cached_x_image import CachedXBlockLOBPCG

            solver_type = CachedXBlockLOBPCG
        if args.active_search_bucket_size and not args.public_api:
            from benchmarks.experimental.active_search import ActiveSearchBlockLOBPCG

            solver_type = ActiveSearchBlockLOBPCG
            solver_options["search_bucket_size"] = args.active_search_bucket_size
            solver_options["keep_history"] = args.active_search_keep_history
        if args.public_api:
            from warpack import BlockLOBPCG as PublicBlockLOBPCG

            solver_type = PublicBlockLOBPCG
            solver_options = dict(
                reuse_operator_images=args.reuse_x_image,
                search_bucket_size=args.active_search_bucket_size,
                keep_history=args.active_search_keep_history,
                preconditioner_bucket_size=args.active_cg_bucket_size,
            )
        starts = wp.array(initial.T.copy(), dtype=wp.float64)
        from benchmarks.experimental.block_kernels import UpstreamSparseOperator
        from warpack import BatchedSparseOperator

        operator = UpstreamSparseOperator if args.native_products else SparseOperator
        if args.batched_products:
            operator = BatchedSparseOperator
        if args.batched_cg:
            from warpack import BatchedCGPreconditioner as BatchedCGInverse

            preconditioner = BatchedCGInverse(
                shifted,
                args.width,
                tol=args.inner_tol,
                atol=0.0,
                maxiter=args.inner_cg,
                preconditioner=pre,
                working_dtype=wp.float32 if args.inner_fp32 else wp.float64,
                fsai_layout=args.fsai_layout,
                matrix_layout=args.matrix_layout,
            )
        elif args.inner_cg:
            preconditioner = CGInverse(
                shifted, tol=args.inner_tol, atol=0.0, maxiter=args.inner_cg, preconditioner=pre
            )
        else:
            preconditioner = UpstreamPreconditioner(pre)
        solver = solver_type(
            operator(a),
            k,
            args.width,
            mass=operator(mass) if mass is not None else None,
            preconditioner=preconditioner,
            initial_vectors=starts,
            constraints=constraints,
            tol=args.tol,
            gram_chunk=args.gram_chunk,
            symmetric_products=args.symmetric_products,
            fused_updates=args.fused_updates,
            paired_rotation=args.paired_rotation,
            **solver_options,
        )
        if args.active_cg_bucket_size and not args.public_api:
            from benchmarks.experimental.active_cg import DeviceBucketedCG

            preconditioner = DeviceBucketedCG(
                preconditioner, solver.errors, solver.tol, args.active_cg_bucket_size
            )
            solver.preconditioner = preconditioner
        return solver, solver.preconditioner

    solver_setup_seconds, (solver, preconditioner) = timed(create_solver)
    capture_seconds, graph = timed(
        lambda: solver.capture(args.steps, adaptive=False if args.fixed_outer else None)
    )
    memory_after = memory_snapshot()
    memory_after_capture = memory_after["device_free_bytes"]
    print("Captured", capture_seconds, flush=True)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dot = Path("build/block-eigen") / (destination.stem + ".dot")
    dot.parent.mkdir(parents=True, exist_ok=True)
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    if not args.keep_graph:
        dot.unlink()
    warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
    report = dict(
        inputs=input_identity,
        dependency=warp_dependencies(type(pre) if args.preconditioner == "fsai" else None),
        args=vars(args),
        n=a.shape[0],
        k=k,
        fixture=fixture,
        # Preserve the historical field's meaning; total setup is explicit.
        setup_seconds=setup_seconds,
        preconditioner_setup_seconds=setup_seconds,
        solver_setup_seconds=solver_setup_seconds,
        total_setup_seconds=setup_seconds + solver_setup_seconds,
        setup_scope=(
            "Preconditioner construction plus solver/operator/CG workspaces, input upload, "
            "and optional FP32 matrix copy. Excludes mesh assembly, reference loading, "
            "host random-start generation, graph capture and warmup. JIT work is included "
            "where encountered; these are not warmed setup measurements. "
            "Historical setup_seconds is an alias for preconditioner_setup_seconds."
        ),
        setup_capture_device_free_memory_decrease_bytes=memory_before_setup - memory_after_capture,
        memory_scope="Device free-memory decrease from after fixture preparation through preconditioner/workspace setup and capture. Includes allocator pools/runtime/JIT allocations; excludes original fixture storage and CuPy adapter. Not peak memory or a pure live-workspace count.",
        memory_snapshots=dict(
            before_setup=memory_before,
            after_preconditioner=memory_after_preconditioner,
            after_capture=memory_after,
        ),
        pool_memory_scope="Warp CUDA pool counters only, excluding driver/context allocations. The high-water counter covers the whole process, including fixture preparation; it is not the setup interval peak. Missing counters are null. Upstream FSAI can consume substantial device memory outside this pool.",
        capture_seconds=capture_seconds,
        warmup_seconds=warmup_seconds,
        graph=audit,
        rows=[],
        versions={
            name: importlib.metadata.version(name)
            for name in ["warp-lang", "numpy", "scipy", "cupy-cuda12x"]
        },
        source_sha256=source_manifest(__file__),
    )
    for _ in range(args.repeats):
        seconds, _ = timed(lambda: wp.capture_launch(graph))
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        quality = check(w, v)
        quality.update(
            status=int(solver.status.numpy()[0]), converged=int(solver.converged.numpy()[0])
        )
        quality["accepted"] &= quality["status"] == 0 and quality["converged"] == k
        if args.inner_cg:
            quality["preconditioner_cg_stats"] = preconditioner.stats.numpy().tolist()
            quality["cg_iteration_accounting"] = (
                "scheduled vector iterations, including masked converged RHSs"
                if args.batched_cg
                else "sum of independently stopped RHS iterations"
            )
        if args.active_cg_bucket_size:
            quality["preconditioner_cg_bucket_widths"] = preconditioner.widths.numpy().tolist()
            quality["preconditioner_cg_bucket_usage"] = preconditioner.usage.numpy().tolist()
            quality["cg_iteration_accounting"] = (
                "Scheduled vector iterations in device-selected buckets, including within-bucket padding; fresh residuals may reactivate rows. Usage stores calls per bucket followed by total active RHSs."
            )
        if args.active_search_bucket_size:
            quality["search_bucket_widths"] = solver.search_widths.numpy().tolist()
            quality["search_bucket_usage"] = solver.search_usage.numpy().tolist()
        report["rows"].append(
            dict(
                method="Warp BlockLOBPCG"
                + (" + device-bucketed CG" if args.active_cg_bucket_size else "")
                + (" + cached A X" if args.reuse_x_image else "")
                + (" + device-bucketed search" if args.active_search_bucket_size else ""),
                seconds=seconds,
                iterations=int(solver.iterations.numpy()[0]),
                quality=quality,
            )
        )
        print(report["rows"][-1], flush=True)
    if args.cupy:
        collect_reference(
            report,
            lambda rows: cupy_trials(
                sa,
                sb,
                pre,
                initial,
                constraints,
                args,
                k,
                check,
                cg=(preconditioner.base if args.active_cg_bucket_size else preconditioner)
                if args.batched_cg
                else None,
                completed_rows=rows,
            )[1],
        )
    if args.profile:
        report["profile"] = profile_step(solver)
        report["post_profile_quality"] = check(
            solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        )
        post = report["post_profile_quality"]
        post.update(
            status=int(solver.status.numpy()[0]), converged=int(solver.converged.numpy()[0])
        )
        post["accepted"] &= post["status"] == 0 and post["converged"] == k
    after = other_gpu_processes()
    report.update(
        other_gpu_processes_before=before,
        other_gpu_processes_after=after,
        timing_eligible=not (before or after),
        timing_note="Warm replay wall times; setup/capture separate. CuPy converges its entire starting block; Warp stops on the first k. Contended or numerically rejected runs are not performance evidence.",
    )
    finalize_acceptance(report)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination, flush=True)
    if not report["accepted"]:
        raise SystemExit("Rejected block comparison; all trial evidence retained")


def profile_step(solver):
    from benchmarks.applications.skinning_profile import Events
    from warpack import block as implementation

    events = Events()
    saved = []
    for obj, attribute, label in (
        (implementation, "gram", "gram"),
        (implementation, "rotate", "rotate"),
        (solver.preconditioner, "apply", "preconditioner"),
        *(
            [(solver.preconditioner, "refresh", "working_matrix_refresh")]
            if hasattr(solver.preconditioner, "refresh")
            else []
        ),
        (solver.op, "apply", "operator"),
        (solver.mass, "apply", "mass"),
        (solver.small, "diagonalize", "rayleigh_ritz_eigh"),
        (solver.metric, "diagonalize", "metric_eigh"),
        *[
            (solver, name, name)
            for name in (
                "project",
                "orthogonalize_x",
                "orthogonalize_search",
                "whiten",
                "finalize",
                "step",
            )
        ],
    ):
        original = getattr(obj, attribute)
        saved.append((obj, attribute, original))
        setattr(obj, attribute, events.wrap(label, original))
    try:
        with wp.ScopedCapture() as capture:
            solver.step()  # Events are legal here: this graph has no conditional body.
    finally:
        for obj, attribute, original in saved:
            setattr(obj, attribute, original)
    samples = []
    for _ in range(5):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        samples.append(events.totals())
    return dict(
        samples_ms=samples,
        medians_ms={key: float(np.median([row[key] for row in samples])) for key in samples[0]},
        note="Instrumented fixed single-step graph, continued from last solve. Nested totals overlap; not additive or equivalent to adaptive whole-solve timings.",
    )


def cupy_trials(a, b, pre, initial, constraints, args, k, check, cg=None, completed_rows=None):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    wp.synchronize()
    setup_start = time.perf_counter()
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        a = cs.csr_matrix(a)
        b = None if b is None else cs.csr_matrix(b)
        width = args.width if args.cupy_width is None else args.cupy_width
        start = cp.asarray(initial[:, :width])
        y = None if constraints is None else cp.asarray(constraints).T
        adapter, adapter_setup_seconds, adapter_memory_bytes = None, 0.0, 0
        adapter_pool_bytes = None
        padded = getattr(args, "cupy_padding", False)
        partitioned = getattr(args, "cupy_partitioned", False)
        if cg is not None:
            from benchmarks.experimental.batched_cg import (
                CapturedActiveCG,
                CapturedPaddedCG,
                CapturedPartitionedCG,
            )

            memory_before = cp.cuda.runtime.memGetInfo()[0]
            pool_before = memory_snapshot()["warp_pool_used_bytes"]
            active_adapter = CapturedPartitionedCG if partitioned else CapturedActiveCG
            adapter_setup_seconds, adapter = timed(
                lambda: CapturedPaddedCG(cg) if padded else active_adapter(cg, max_width=width)
            )
            adapter_memory_bytes = memory_before - cp.cuda.runtime.memGetInfo()[0]
            pool_after = memory_snapshot()["warp_pool_used_bytes"]
            if pool_before is not None and pool_after is not None:
                adapter_pool_bytes = pool_after - pool_before

        def apply(x):
            one = x.ndim == 1
            rows = cp.ascontiguousarray(x.reshape(-1, 1).T if one else x.T)
            out = cp.empty_like(rows)
            wx, wy = wp.from_dlpack(rows), wp.from_dlpack(out)
            if adapter is None:
                for i in range(len(rows)):
                    pre.matvec(wx[i], wy[i], wy[i], 1.0, 0.0)
            else:
                adapter.apply(wx, wy)
            return out[0] if one else out.T

        preconditioner = cl.LinearOperator(a.shape, matvec=apply, matmat=apply, dtype=cp.float64)
        stream.synchronize()
        reference_setup_seconds = time.perf_counter() - setup_start
        cupy_tol = args.cupy_tol
        if cupy_tol is None:
            cupy_tol = 1e-5 if args.case == "anvil" else 1e-10

        def solve():
            if cg is not None:
                if padded:
                    cg.reset_stats()
                else:
                    adapter.reset_stats()
            return cl.lobpcg(
                a,
                start.copy(),
                B=b,
                M=preconditioner,
                Y=y,
                largest=False,
                tol=cupy_tol,
                maxiter=args.steps,
                retResidualNormsHistory=True,
            )

        solve()
        rows = []
        for _ in range(args.repeats):
            seconds, (w, v, history) = timed(solve)
            quality = check(w[:k].get(), v[:, :k].get())
            # CuPy's history can end before its last Ritz update; it is not
            # the actual residual of the returned vectors. Check fresh products.
            actual_residuals = cp.linalg.norm(a @ v - (v if b is None else b @ v) * w, axis=0)
            absolute_residuals = actual_residuals.get().tolist()
            if cg is not None:
                stats = cg.stats if padded else adapter.collect_stats()
                quality["preconditioner_cg_stats"] = stats.numpy().tolist()
                quality["cg_iteration_accounting"] = (
                    "scheduled vector iterations, including masked RHSs"
                    + (" and padded zero RHSs" if padded else "; no padding")
                    + ("; independently stopped power-of-two chunks" if partitioned else "")
                )
                if not padded:
                    quality["cg_stats_by_active_width"] = {
                        str(width): item.inverse.stats.numpy().tolist()
                        for width, item in adapter.adapters.items()
                    }
            rows.append(
                dict(
                    method=(
                        "CuPy LOBPCG + same captured upstream Warp batched CG"
                        + (" (power-of-two chunks)" if partitioned else "")
                        if cg is not None
                        else "CuPy LOBPCG + same upstream Warp preconditioner"
                    ),
                    seconds=seconds,
                    residual_history_length=len(history),
                    last_history_absolute_residuals=history[-1].get().tolist(),
                    returned_absolute_residuals=absolute_residuals,
                    returned_k_absolute_tolerance_met=max(absolute_residuals[:k]) <= cupy_tol,
                    full_block_absolute_tolerance_met=max(absolute_residuals) <= cupy_tol,
                    quality=quality,
                )
            )
            if completed_rows is not None:
                completed_rows.append(rows[-1])
            print(rows[-1], flush=True)
        return rows, dict(
            width=width,
            absolute_tolerance=cupy_tol,
            padding_width=cg.width if cg is not None and padded else None,
            preallocated_active_widths=list(adapter.adapters)
            if cg is not None and not padded
            else None,
            active_partition="power-of-two exact chunks" if partitioned else "whole active block",
            preallocated_rhs_slots=sum(adapter.adapters)
            if cg is not None and not padded
            else (cg.width if cg is not None else 0),
            inner_adapter_setup_capture_seconds=adapter_setup_seconds,
            reference_setup_seconds=reference_setup_seconds,
            reference_setup_scope="CuPy matrix/start/constraint upload, LinearOperator, and inner adapter workspaces/capture. Excludes supplied FSAI/base CG construction, library imports, and outer eigensolve warmup.",
            inner_adapter_device_memory_delta_bytes=adapter_memory_bytes,
            inner_adapter_warp_pool_used_delta_bytes=adapter_pool_bytes,
            pool_memory_scope="Process-local Warp pool used-memory delta during adapter construction; excludes graph/driver and non-pool allocations. Not peak memory.",
            memory_scope="Device free-memory decrease during adapter setup, including graph/runtime allocations; not a peak measurement.",
            setup_scope="Extra inner adapter workspace/capture only. Shifted matrix and FSAI factors are shared with the completed Warp trial; active-width CG states are preallocated before timing. This is not independent end-to-end CuPy setup.",
            stopping="CuPy uses absolute residual tolerance and converges its entire starting block; independent original-problem gates check the first k modes. Warp uses normalized stopping on k modes.",
        )


if __name__ == "__main__":
    main()
