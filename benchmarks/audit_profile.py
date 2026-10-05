"""Audit graph replay overhead and restart costs on the published large matrix.

Uses CUDA events, not an API tracer. Stage timings are a separate instrumented
restart and include event-node overhead. Inputs are produced by compare.py.
"""

import argparse
import collections
import json
import re
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csl
import cupyx.scipy.sparse.linalg._eigen as cupy_eigen
import numpy as np
import scipy.io
import warp as wp

from benchmarks.compare import quality, wp_matrix
from warpack import KrylovSchur, SparseOperator


def graph_summary(path):
    text = Path(path).read_text()
    return {
        "node_types_including_conditional_body": dict(
            collections.Counter(re.findall(r'label="\{\s*([A-Z_]+)', text))
        ),
        "memcpy_kinds": dict(collections.Counter(re.findall(r"\| \{kind \| ([^}]+)", text))),
        "method": "CUDA graph DOT inspection; not a CUDA API trace",
    }


def replay_timing(graph, repeats=5):
    wp.capture_launch(graph)
    wp.synchronize()
    begin = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    rows = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        wp.record_event(begin)
        event_start_done = time.perf_counter()
        wp.capture_launch(graph)
        launch_done = time.perf_counter()
        wp.record_event(end)
        t1 = time.perf_counter()
        wp.synchronize()
        t2 = time.perf_counter()
        rows.append(
            [
                1000 * (t1 - t0),
                wp.get_event_elapsed_time(begin, end, False),
                1000 * (t2 - t0),
                1000 * (launch_done - event_start_done),
                1000 * (t1 - launch_done),
            ]
        )
    return dict(
        zip(
            [
                "host_enqueue_ms",
                "device_elapsed_ms",
                "wall_ms",
                "graph_launch_call_ms",
                "end_event_call_ms",
            ],
            np.median(rows, axis=0),
        )
    )


def profile_restart(s):
    """Use one representative restart after the solver has converged."""
    events, stages = [], []

    def mark(name):
        event = wp.Event(enable_timing=True)
        events.append(event)
        stages.append(name)
        wp.record_event(event, external=True)

    with wp.ScopedCapture() as cap:
        mark("start")
        for j in range(s.keep, s.m):
            s.expand(j)
        mark("expand_and_reorthogonalize")
        s.op.apply(s.views[-1], s.zviews[-1])
        mark("last_operator_product")
        s.complete_projection() if hasattr(s, "complete_projection") else s.gram(s.q, s.z)
        mark("projected_gram")
        s.diagonalize()
        mark("projected_eigensolve")
        s.rotate(s.q, s.retained)
        mark("rotate_basis")
        s.update_retained_products() if hasattr(s, "update_retained_products") else s.rotate(
            s.z, s.retained_ax
        )
        mark("rotate_operator_products")
        wp.copy(s.q, s.retained, count=s.keep * s.n)
        wp.copy(s.z, s.retained_ax, count=s.keep * s.n)
        s.finalize()
        if hasattr(s, "reset_projection"):
            s.reset_projection()
        mark("copy_and_residual")
    samples = []
    for _ in range(6):
        wp.capture_launch(cap.graph)
        wp.synchronize()
        samples.append(
            [wp.get_event_elapsed_time(a, b, False) for a, b in zip(events[:-1], events[1:])]
        )
    return dict(zip(stages[1:], np.median(samples[1:], axis=0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="build/heterogeneous_laplacian_125000.mtx")
    p.add_argument("--out", default="results/audit_profile.json")
    p.add_argument("--same-start", action="store_true")
    args = p.parse_args()
    a = scipy.io.mmread(args.matrix).tocsr()
    op = SparseOperator(wp_matrix(a))
    report = {
        "n": a.shape[0],
        "k": 20,
        "ncv": 48,
        "warp": [],
        "cupy": [],
        "shared_starting_vector": args.same_start,
        "starting_vector_reset_per_trial": True,
    }
    starting_vector = None
    for tol in [1e-9, 1e-10, 1e-11]:
        s = KrylovSchur(op, 20, ncv=48, which="LA", tol=tol)
        graph = s.capture(100)
        if args.same_start and starting_vector is None:
            s.initialize()
            starting_vector = cp.asarray(s.views[0].numpy().ravel())
        row = {
            "tol": tol,
            **replay_timing(graph),
            "restarts": int(s.iterations.numpy()[0]),
            "converged": int(s.converged.numpy()[0]),
            **quality(a, s.eigenvalues.numpy(), s.eigenvectors.numpy().T),
        }
        row.pop("eigenvalues", None)
        report["warp"].append(row)
        if tol == 1e-11:
            Path("build/audit").mkdir(exist_ok=True)
            wp.capture_debug_dot_print(graph, "build/audit/large_graph.dot")
            report["graph_audit"] = graph_summary("build/audit/large_graph.dot")
            report["restart_stage_ms"] = profile_restart(s)
            s.solve(1)
            wp.synchronize()
            with wp.ScopedTimer(
                "uncaptured_restart_kernels", cuda_filter=wp.TIMING_KERNEL, print=False
            ) as timer:
                s.step()
            grouped = {}
            for result in timer.timing_results:
                name = result.name
                grouped[name] = grouped.get(name, 0.0) + result.elapsed
            report["instrumented_uncaptured_kernel_ms"] = grouped
    d = csp.csr_matrix(a)
    orig_lanczos, orig_ritz = cupy_eigen._lanczos_fast, cupy_eigen._eigsh_solve_ritz
    counter = {}

    def lanczos_factory(*args):
        original = orig_lanczos(*args)

        def run(*args):
            counter["lanczos_matvecs"] += args[-1] - args[-2]
            return original(*args)

        return run

    def ritz(*args):
        counter["ritz_solves"] += 1
        return orig_ritz(*args)

    cupy_eigen._lanczos_fast, cupy_eigen._eigsh_solve_ritz = lanczos_factory, ritz
    try:
        for tol in [1e-9, 1e-10]:
            samples = []
            for _ in range(4):
                counter.update(lanczos_matvecs=0, ritz_solves=0)
                cp.random.seed(42)
                v0 = starting_vector.copy() if starting_vector is not None else None
                cp.cuda.get_current_stream().synchronize()
                t0 = time.perf_counter()
                w, v = csl.eigsh(d, k=20, ncv=48, which="LA", tol=tol, maxiter=10000, v0=v0)
                cp.cuda.get_current_stream().synchronize()
                samples.append(1000 * (time.perf_counter() - t0))
            report["cupy"].append(
                {
                    "tol": tol,
                    "wall_ms": float(np.median(samples[1:])),
                    "ritz_solves": counter["ritz_solves"],
                    "matvecs": counter["lanczos_matvecs"] + counter["ritz_solves"] - 1,
                    **quality(a, cp.asnumpy(w), cp.asnumpy(v)),
                }
            )
    finally:
        cupy_eigen._lanczos_fast, cupy_eigen._eigsh_solve_ritz = orig_lanczos, orig_ritz
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
