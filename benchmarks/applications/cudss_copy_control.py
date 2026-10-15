"""Diagnostic only: substitute device sources for fixed vendor metadata bytes.

This modifies private cuDSS graph nodes and is not a supported inverse backend.
It measures a fixed captured pencil; source changes/refactorization are outside
this experiment's contract. Production Warpack never imports this helper.
"""

import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from cuda.bindings import runtime as cuda

from .compare import other_gpu_processes
from .cudss_large_audit import measure_enqueue
from .cudss_scalar_probe import checked, copies_from_dot
from .fem_cases import make_fem_case
from .nullspace_compare import audit
from .runner import accuracy, warp_output, warp_setup


def device_sources(graph, path):
    if graph.graph_exec is not None:
        raise ValueError("Modify only the uninstantiated diagnostic graph")
    sources = {}
    nodes = copies_from_dot(path)
    for row in nodes:
        if row["bytes"] != 8:
            raise ValueError("Diagnostic expects only the observed eight-byte metadata copies")
        source = row["source"]
        if source not in sources:
            data = np.frombuffer(bytes.fromhex(row["payload_hex"]), dtype=np.uint32).copy()
            sources[source] = wp.array(data, dtype=wp.uint32)
        (params,) = checked(cuda.cudaGraphMemcpyNodeGetParams(int(row["node"], 16)))
        if int(params.srcPtr.ptr) != int(source, 16):
            raise ValueError("DOT source and live graph disagree")
        params.srcPtr.ptr = sources[source].ptr
        params.kind = cuda.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        checked(cuda.cudaGraphMemcpyNodeSetParams(int(row["node"], 16), params))
    return sources, nodes


def main():
    folder = Path("build/cudss-graph-audit")
    p = make_fem_case("clamped_consistent", side=10).problem
    solver, inverse = warp_setup(p, ncv=32)
    original = solver.capture(1)
    control = solver.capture(1)
    report = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        diagnostic_only=True,
        supported_production_backend=False,
        scope="Fixed captured 10890-DOF FEM pencil, unchanged factors/configuration/host metadata. Private vendor nodes edited for a causal source-residency control only.",
        n=p.n,
        k=p.k,
        ncv=32,
        steps=1,
        original_audit=audit(original, folder / "control-original.dot"),
    )
    audit(control, folder / "control-before.dot")
    sources, nodes = device_sources(control, folder / "control-before.dot")
    report["replaced_nodes"] = nodes
    report["device_source_buffers"] = len(sources)
    report["control_audit"] = audit(control, folder / "control-after.dot")

    def validate():
        check = accuracy(p, *warp_output(p, solver))
        check["count"] = int(solver.converged.numpy()[0])
        check["status"] = int(solver.status.numpy()[0])
        check["accepted"] &= check["count"] == p.k and check["status"] == 0
        return check

    try:
        report["other_gpu_processes_before"] = other_gpu_processes()
        report["original"] = measure_enqueue(original, validate)
        report["device_source_control"] = measure_enqueue(control, validate)
        report["other_gpu_processes_after"] = other_gpu_processes()
        report["all_individually_completed_trials_accepted"] = all(
            r["validation"]["accepted"]
            for key in ["original", "device_source_control"]
            for r in report[key]["single_replays"]
        )
        report["all_burst_final_outputs_accepted"] = all(
            report[key]["queued_burst_diagnostic"]["final_validation"]["accepted"]
            for key in ["original", "device_source_control"]
        )
        (folder / "copy-control.json").write_text(json.dumps(report, indent=2) + "\n")
        for key in ["original", "device_source_control"]:
            print(
                key,
                [
                    (r["enqueue_seconds"], r["completion_seconds"])
                    for r in report[key]["single_replays"]
                ],
                report[key]["queued_burst_diagnostic"]["enqueue_seconds"],
                flush=True,
            )
        if (
            not report["all_individually_completed_trials_accepted"]
            or not report["all_burst_final_outputs_accepted"]
            or report["other_gpu_processes_before"]
            or report["other_gpu_processes_after"]
        ):
            raise SystemExit("Diagnostic control accuracy or contention failure")
    finally:
        original = control = solver.graph = None
        gc.collect()
        inverse.close()
        sources.clear()


if __name__ == "__main__":
    main()
