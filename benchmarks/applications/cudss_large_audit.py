"""Matched large FEM settings and full-graph API trace; instrumentation is separate."""

import argparse
import gc
import hashlib
import json
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CGInverse

from .compare import compare_baseline, run
from .cuda_api_trace import ApiTrace
from .cudss_config_probe import PROFILES, config_values, configuration
from .cudss_scalar_probe import copies_from_dot
from .fem_cases import make_fem_case
from .nullspace_compare import audit
from .runner import accuracy, warp_output, warp_setup


def trace_graph(trace, graph, label):
    # Instantiate/upload/warm outside the traced replay region.
    wp.capture_launch(graph)
    wp.synchronize()
    trace.phase = label + ":warm_replay"
    for _ in range(3):
        wp.capture_launch(graph)
    trace.phase = label + ":caller_sync"
    wp.synchronize()
    trace.phase = None


def measure_enqueue(graph, validate):
    """Uninstrumented launch vs completion latency, with explicit caller waits."""
    wp.capture_launch(graph)
    wp.synchronize()
    rows = []
    for _ in range(3):
        start = time.perf_counter()
        wp.capture_launch(graph)
        enqueued = time.perf_counter()
        wp.synchronize()
        complete = time.perf_counter()
        rows.append(
            dict(
                enqueue_seconds=enqueued - start,
                completion_seconds=complete - start,
                validation=validate(),
            )
        )
    wp.synchronize()
    start = time.perf_counter()
    launches = []
    for _ in range(3):
        begin = time.perf_counter()
        wp.capture_launch(graph)
        launches.append(time.perf_counter() - begin)
    wp.synchronize()
    burst = dict(
        enqueue_seconds=launches,
        completion_seconds=time.perf_counter() - start,
        final_validation=validate(),
        intermediate_outputs_validated=False,
    )
    return dict(single_replays=rows, queued_burst_diagnostic=burst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-timings", action="store_true")
    args = parser.parse_args()
    folder = Path("build/cudss-graph-audit")
    folder.mkdir(parents=True, exist_ok=True)
    p = make_fem_case("clamped_consistent", side=10).problem
    report = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        timings=[],
        trace=[],
    )
    if args.skip_timings:
        previous = json.loads((folder / "large-audit.json").read_text())
        digest = hashlib.sha256()
        for mat in (p.a, p.b):
            for data in (mat.indptr, mat.indices, mat.data):
                digest.update(data.tobytes())
        first = previous["timings"][0]
        if (
            previous["gpu"],
            previous["warp"],
            first["matrix_sha256"],
            first["target"],
            first["k"],
        ) != (report["gpu"], report["warp"], digest.hexdigest(), [p.target, 0.0], p.k):
            raise ValueError("Saved timing environment or pencil does not match the trace")
        report = previous
    report["trace_source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report["versions"] = {
        n: version(n) for n in ("cuda-bindings", "nvidia-cuda-cupti-cu12", "nvidia-cudss-cu12")
    }

    for profile in [] if args.skip_timings else PROFILES:
        # Do not enable CUPTI during timing comparisons.
        with configuration(profile):
            row = run("A", p.n, "warp", repeats=3, ncv=32, steps=1, problem=p)
        row["configuration_profile"] = profile
        report["timings"].append(row)
        print(profile, row["outcome"], row.get("solve_seconds"), flush=True)
        (folder / "large-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    baseline = json.loads(Path("results/fem_side10_krylov32_cycles1.json").read_text())
    report["saved_baseline_checks"] = compare_baseline([report["timings"][0]], baseline, 1.25)
    solver, inverse = warp_setup(p, ncv=32)
    graph = solver.capture(1)
    try:
        report["full_solver_config"] = config_values(inverse.solver)
        report["full_solver_graph_audit"] = audit(graph, folder / "full-fem.dot")
        report["full_solver_copies"] = copies_from_dot(folder / "full-fem.dot")
        report["full_solver_enqueue"] = measure_enqueue(
            graph, lambda: accuracy(p, *warp_output(p, solver))
        )
        with ApiTrace() as trace:
            trace_graph(trace, graph, "fem_cudss")
        events, errors = trace.events, trace.errors
        report["full_solver_accuracy"] = accuracy(p, *warp_output(p, solver))
        report["full_solver_count"] = int(solver.converged.numpy()[0])
    finally:
        graph = solver.graph = None
        gc.collect()
        inverse.close()
    # A purely Warp inverse supplies a graph with no host transfers as a control.
    n = 256
    a = sp.diags([[-0.1] * (n - 1), [2.0] * n, [-0.1] * (n - 1)], [-1, 0, 1], format="csr")
    inverse = CGInverse(matrix(a), tol=1e-13, maxiter=100, preconditioner="diag")
    rhs = wp.ones((1, n), dtype=wp.float64)
    inverse.apply(rhs, inverse.x.reshape((1, n)))
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        inverse.apply(rhs, inverse.x.reshape((1, n)))
    report["pure_warp_graph_audit"] = audit(cap.graph, folder / "pure-warp-cg.dot")
    report["pure_warp_enqueue"] = measure_enqueue(
        cap.graph, lambda: dict(accepted=bool(abs(a @ inverse.x.numpy() - 1).max() < 1e-11))
    )
    with ApiTrace() as trace:
        trace_graph(trace, cap.graph, "pure_warp_cg")
    events += trace.events
    errors += trace.errors
    output = inverse.x.numpy()
    report["pure_warp_residual"] = float(abs(a @ output - 1).max())
    report["trace_errors"] = errors
    report["trace"] = events
    report["api_entries_by_phase"] = {
        phase: dict(Counter(e["name"] for e in events if e["site"] == 0 and e["phase"] == phase))
        for phase in sorted({e["phase"] for e in events})
    }
    (folder / "large-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    forbidden_replay_calls = [
        e
        for e in events
        if e["phase"].endswith(":warm_replay")
        and ("Synchronize" in e["name"] or "Memcpy" in e["name"])
    ]
    if (
        forbidden_replay_calls
        or any(
            not row["validation"]["accepted"]
            for key in ("full_solver_enqueue", "pure_warp_enqueue")
            for row in report[key]["single_replays"]
        )
        or any(e.get("status", 0) for e in events)
        or report["full_solver_count"] != p.k
        or errors
        or not report["full_solver_accuracy"]["accepted"]
        or report["pure_warp_residual"] > 1e-11
        or any(
            not r["accepted"]
            or not r["timing_uncontended"]
            or any(not v["solver_full_count"] for v in r["validation"])
            for r in report["timings"]
        )
        or not any(c["comparable"] for c in report["saved_baseline_checks"])
        or any(
            c.get("performance_regression")
            or c.get("accuracy_regression")
            or c.get("graph_regression")
            for c in report["saved_baseline_checks"]
        )
    ):
        raise SystemExit("Audit validation failed")
    print(report["saved_baseline_checks"], report["api_entries_by_phase"], flush=True)


if __name__ == "__main__":
    main()
