"""Diagnostic only: substitute cuDSS graph's captured scalar-copy sources.

This examines captured internals, not a supported production workaround. A
future cuDSS release can change the meaning or lifetime of these host buffers.
"""

import ctypes
import json
from pathlib import Path

import numpy as np
import warp as wp
from cuda.bindings import runtime as rt

from benchmarks.audit_profile import replay_timing
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator
from warpack.fem import assemble_rest, mass_normalized, read_mesh


def check(result):
    assert int(result[0]) == 0, result
    return result[1:]


def replace_sources(graph, where):
    wp.synchronize()
    if graph.graph_exec is not None:
        check(rt.cudaGraphExecDestroy(graph.graph_exec.value))
        graph.graph_exec = None
    _, count = check(rt.cudaGraphGetNodes(graph.graph.value))
    nodes, _ = check(rt.cudaGraphGetNodes(graph.graph.value, count))
    buffers, changed, values = {}, 0, []
    for node in nodes:
        (kind,) = check(rt.cudaGraphNodeGetType(node))
        if kind != rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            continue
        (params,) = check(rt.cudaGraphMemcpyNodeGetParams(node))
        if params.kind != rt.cudaMemcpyKind.cudaMemcpyHostToDevice:
            continue
        assert params.extent.width == 8 and params.extent.height == 1 and params.extent.depth == 1
        ptr = int(params.srcPtr.ptr)
        if ptr not in buffers:
            data = np.frombuffer(ctypes.string_at(ptr, 8), dtype=np.uint8).copy()
            values.append(data.tobytes().hex())
            buffers[ptr] = wp.array(
                data,
                dtype=wp.uint8,
                device="cpu" if where == "pinned" else "cuda:0",
                pinned=where == "pinned",
            )
        copy_kind = (
            rt.cudaMemcpyKind.cudaMemcpyHostToDevice
            if where == "pinned"
            else rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        )
        check(
            rt.cudaGraphMemcpyNodeSetParams1D(
                node, int(params.dstPtr.ptr), buffers[ptr].ptr, 8, copy_kind
            )
        )
        changed += 1
    wp.synchronize()
    return list(buffers.values()), {"copies_changed": changed, "unique_source_values_hex": values}


def main():
    x, t = read_mesh("../bbw-comparison/dragon-H/dragon.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    inverse = CuDSSInverse(mass_normalized(h, mass, shift=1.0), 1)
    solver = KrylovSchur(inverse, 26, ncv=48, which="LA", tol=1e-12)
    evaluation = EigenpairEvaluation(SparseOperator(a), solver.eigenvectors, tol=1e-7)
    graph = solver.capture(3, adaptive=False, finalizer=evaluation.run)
    rows, keep_alive = [], []
    reference = None
    for variant in ["original", "pinned", "device"]:
        notes = {}
        if variant != "original":
            buffers, notes = replace_sources(graph, variant)
            keep_alive.extend(buffers)
        row = {
            "variant": variant,
            **notes,
            **replay_timing(graph, repeats=3),
            "residual": float(evaluation.residuals.numpy().max()),
        }
        vals = evaluation.values.numpy()
        if reference is None:
            reference = vals
        row["max_eigenvalue_change"] = float(np.max(abs(vals - reference)))
        rows.append(row)
        print(row, flush=True)
    wp.capture_debug_dot_print(graph, "build/audit/dragon_device_constants.dot")
    Path("results/audit_cudss_copies.json").write_text(json.dumps(rows, indent=2) + "\n")
    inverse.close()


if __name__ == "__main__":
    main()
