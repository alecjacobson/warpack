"""Diagnostic event profiles of actual active-search branches at fixed checkpoints.

Host selection is for profiling only. The uninstrumented solver remains fully
device-controlled; no numerical implementation or stopping policy is changed.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from warp.optim import linear

from benchmarks.block_inputs import initial_block
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.compare_block import prepare
from benchmarks.graph_audit import graph_summary
from benchmarks.source_provenance import source_manifest
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG
from warpack import block as implementation
from warpack import kernels as K
from warpack._block_search import select_search


class Events:
    """External records wrap selected stages, never CUDA conditional bodies."""

    def __init__(self):
        self.pairs = []
        self.suspended = 0

    def wrap(self, name, function):
        def call(*args, **kwargs):
            if self.suspended:
                return function(*args, **kwargs)
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start, external=True)
            result = function(*args, **kwargs)
            wp.record_event(end, external=True)
            self.pairs.append((name, start, end))
            return result

        return call

    def totals(self):
        totals = defaultdict(float)
        for name, start, end in self.pairs:
            totals[name] += wp.get_event_elapsed_time(start, end, False)
        return dict(totals)


@wp.kernel
def next_prefix_step(count: wp.array[wp.int32], running: wp.array[wp.int32], limit: int):
    count[0] += 1
    running[0] = int(count[0] < limit)


def create_solver(config):
    required = ("public_api", "batched_products", "batched_cg", "reuse_x_image")
    if not all(getattr(config, name) for name in required) or config.preconditioner != "fsai":
        raise ValueError("profile requires the public batched FSAI/CG composition with image reuse")
    if config.active_search_bucket_size <= 0 or config.native_products:
        raise ValueError("profile requires active search and batched products")
    fixture = {}
    a, mass, shifted, constraints, host_a, host_m, k, check = prepare(config, metadata=fixture)
    shifted.nnz_sync()
    factor = linear.FSAI(
        shifted, max_row_size=config.fsai_width, max_step_size=3, factor_dtype=wp.float32
    )
    pre = BatchedCGPreconditioner(
        shifted,
        config.width,
        tol=config.inner_tol,
        atol=0.0,
        maxiter=config.inner_cg,
        preconditioner=factor,
        working_dtype=wp.float32 if config.inner_fp32 else wp.float64,
        fsai_layout=config.fsai_layout,
        matrix_layout=config.matrix_layout,
    )
    # Honor width-control pools while preserving legacy configurations/starts.
    starts = initial_block(
        a.shape[0], config.width, getattr(config, "initial_pool_width", None)
    ).T.copy()
    solver = BlockLOBPCG(
        BatchedSparseOperator(a),
        k,
        config.width,
        mass=None if mass is None else BatchedSparseOperator(mass),
        constraints=constraints,
        preconditioner=pre,
        initial_vectors=wp.array(starts, dtype=wp.float64),
        tol=config.tol,
        gram_chunk=config.gram_chunk,
        symmetric_products=config.symmetric_products,
        fused_updates=config.fused_updates,
        paired_rotation=config.paired_rotation,
        reuse_operator_images=True,
        search_bucket_size=config.active_search_bucket_size,
        keep_history=config.active_search_keep_history,
        preconditioner_bucket_size=config.active_cg_bucket_size,
    )
    digest = hashlib.sha256()
    for mat in (host_a, host_m):
        if mat is not None:
            for array in (mat.indptr, mat.indices, mat.data):
                digest.update(array.tobytes())
    digest.update(starts.tobytes())
    return solver, check, fixture, digest.hexdigest()


def select_branch(solver):
    search = solver._search
    wp.launch(
        select_search,
        1,
        [solver.errors, solver.tol, search.rows, search.count, search.widths, search.flags],
        device=solver.device,
    )
    flags, count = search.flags.numpy(), int(search.count.numpy()[0])
    selected = np.flatnonzero(flags)
    if len(selected) != 1:
        raise ValueError("expected exactly one active search branch")
    return int(selected[0]), count


def capture_profile(solver, index=None):
    search = solver._search
    if (search is None) != (index is None):
        raise ValueError("provide a branch index exactly when active search is enabled")
    branch = solver if search is None else search.buckets[index]
    events, saved = Events(), []
    wrappers = [
        (branch, "gram", "gram"),
        (implementation, "rotate", "rotate"),
        (solver.preconditioner, "apply", "preconditioner"),
        (branch.op, "apply", "operator"),
        (branch.mass, "apply", "mass"),
        (branch.small, "diagonalize", "rayleigh_ritz_eigh"),
        (branch.metric, "diagonalize", "metric_eigh"),
    ] + [
        (branch, name, name)
        for name in (
            "project",
            "orthogonalize_x",
            "orthogonalize_search",
            "whiten",
            "finalize",
            "rayleigh_ritz",
        )
    ]

    def step():
        if search is None:
            solver.step()
            return
        wp.launch(
            select_search,
            1,
            [solver.errors, solver.tol, search.rows, search.count, search.widths, search.flags],
            device=solver.device,
        )
        solver.preconditioner.apply(solver.r, search.correction)
        search.step_bucket(branch, index)

    try:
        from warpack.block_products import (
            paired_ritz_rotation,
            subtract_projection,
            wide_paired_ritz_rotation,
        )

        original_tiled = wp.launch_tiled

        def instrument_tiled(kernel, *args, **kwargs):
            if kernel in (paired_ritz_rotation, wide_paired_ritz_rotation):
                return events.wrap("paired_ritz_rotation", original_tiled)(kernel, *args, **kwargs)
            if kernel is subtract_projection:
                return events.wrap("subtract_projection", original_tiled)(kernel, *args, **kwargs)
            return original_tiled(kernel, *args, **kwargs)

        saved.append((wp, "launch_tiled", original_tiled))
        wp.launch_tiled = instrument_tiled
        for obj, attribute, label in wrappers:
            original = getattr(obj, attribute)
            saved.append((obj, attribute, original))
            setattr(obj, attribute, events.wrap(label, original))
        for attribute, label in (
            ("_ritz_qr_reuse", "ritz_metric_qr"),
            ("_first_whitener", "first_metric_whitening"),
            ("_second_whitener", "second_metric_whitening"),
        ):
            helper = getattr(branch, attribute, None)
            if helper is not None:
                original_helper = helper.apply

                def conditional_helper(*args, _apply=original_helper, **kwargs):
                    # External CUDA event nodes cannot live inside conditional
                    # child graphs. Time the complete helper at its boundary.
                    events.suspended += 1
                    try:
                        return _apply(*args, **kwargs)
                    finally:
                        events.suspended -= 1

                saved.append((helper, "apply", original_helper))
                helper.apply = events.wrap(label, conditional_helper)
        original_launch = branch.launch

        def instrument_launch(kernel, shape, arguments):
            if kernel in (K.cholesky, K.triangular):
                name = "cholesky" if kernel == K.cholesky else "triangular"
                return events.wrap(name, original_launch)(kernel, shape, arguments)
            return original_launch(kernel, shape, arguments)

        saved.append((branch, "launch", original_launch))
        branch.launch = instrument_launch
        with wp.ScopedCapture(device=solver.device) as capture:
            events.wrap("selected_step", step)()
    finally:
        for obj, attribute, original in reversed(saved):
            setattr(obj, attribute, original)
    return capture.graph, events


def checked(solver, check):
    result = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
    result["status"] = int(solver.status.numpy()[0])
    result["converged"] = int(solver.converged.numpy()[0])
    result["iterations"] = int(solver.iterations.numpy()[0])
    result["accepted"] &= result["status"] == 0 and result["converged"] == solver.k
    return result


def profile(solver, check, steps, checkpoints, repeats):
    full = solver.capture(steps)
    wp.capture_launch(full)
    before = checked(solver, check)
    if not before["accepted"]:
        raise ValueError("uninstrumented full solve failed before profiling")
    if any(c < 0 or c >= before["iterations"] for c in checkpoints):
        raise ValueError("checkpoints must precede converged full-solve iteration count")
    records, profiles = [], {}
    prefix_count = wp.zeros(1, dtype=wp.int32, device=solver.device)
    prefix_running = wp.zeros_like(prefix_count)
    for checkpoint in checkpoints:

        def body():
            solver.step()
            wp.launch(
                next_prefix_step,
                1,
                [prefix_count, prefix_running, checkpoint],
                device=solver.device,
            )

        with wp.ScopedCapture(device=solver.device) as capture:
            solver.initialize()
            prefix_count.zero_()
            if checkpoint:
                prefix_running.fill_(1)
                wp.capture_while(prefix_running, body)
        prefix = capture.graph
        for trial in range(repeats):
            wp.capture_launch(prefix)
            index, count = select_branch(solver)
            branch = solver._search.buckets[index]
            if index not in profiles:
                profiles[index] = capture_profile(solver, index)
            graph, events = profiles[index]
            wp.capture_launch(graph)
            wp.synchronize_device(solver.device)
            if solver.status.numpy()[0] != 0:
                raise ValueError("instrumented branch produced nonzero status")
            records.append(
                dict(
                    checkpoint=checkpoint,
                    trial=trial,
                    active_rows=count,
                    bucket_index=index,
                    search_width=int(solver.search_widths.numpy()[index]),
                    projected_dimension=branch.m,
                    metric_dimension=branch.metric.m,
                    events_ms=events.totals(),
                    projected_sweeps=None
                    if branch.small._global_jacobi is None
                    else int(branch.small._global_jacobi.sweeps.numpy()[0]),
                    metric_sweeps_scope="Last executed canonical metric solve, possibly from an earlier step when both certificates succeed. metric_eigh events exclude all conditional helper work; first_metric_whitening and second_metric_whitening each include their complete helper. These spans overlap enclosing whiten/stage events and must not be added.",
                    metric_sweeps=None
                    if branch.metric._global_jacobi is None
                    else int(branch.metric._global_jacobi.sweeps.numpy()[0]),
                )
            )
    wp.capture_launch(full)
    after = checked(solver, check)
    if not after["accepted"]:
        raise ValueError("uninstrumented full solve failed after profiling")
    return dict(before=before, after=after, samples=records), full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("results/cg_partitioned/final/exact-a1.json")
    )
    parser.add_argument("--checkpoints", default="1,10,20,30")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeat count required")
    config = SimpleNamespace(**json.loads(args.config.read_text())["args"])
    start = time.perf_counter()
    solver, check, fixture, identity = create_solver(config)
    setup = time.perf_counter() - start
    report, graph = profile(
        solver, check, config.steps, list(map(int, args.checkpoints.split(","))), args.repeats
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dot = args.out.with_suffix(".dot")
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    dot.unlink()
    assert not audit["node_types_including_conditional_body"].get("HOST", 0)
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
    report.update(
        args=vars(args) | dict(config=str(args.config), out=str(args.out)),
        solver_config=vars(config),
        n=solver.n,
        k=solver.k,
        width=solver.b,
        fixture=fixture,
        input_sha256=identity,
        setup_seconds=setup,
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        graph=audit,
        accepted=report["before"]["accepted"] and report["after"]["accepted"],
        protocol="Diagnostic fixed-prefix replays followed by the branch selected from that state. Host readback selects profiling instrumentation only. Nested event totals overlap and are not additive; not adaptive whole-solve timings. The uninstrumented device-only full solve must pass original physical/count/subspace gates before and after profiling.",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for c in sorted({r["checkpoint"] for r in report["samples"]}):
        rows = [r for r in report["samples"] if r["checkpoint"] == c]
        print(
            c,
            [(r["active_rows"], r["search_width"]) for r in rows],
            {
                key: float(np.median([r["events_ms"][key] for r in rows]))
                for key in rows[0]["events_ms"]
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
