"""Inspect vendor HtoD scalar nodes and separate replay API calls from host waits."""

import ctypes as ct
import gc
import json
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
from cuda.bindings import runtime as cuda

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse

from .cuda_api_trace import ApiTrace


def checked(result):
    if int(result[0]):
        raise RuntimeError(result[0])
    return result[1:]


def copies(graph):
    handle = getattr(graph.graph, "value", graph.graph)
    _, count = checked(cuda.cudaGraphGetNodes(handle))
    nodes, _ = checked(cuda.cudaGraphGetNodes(handle, count))
    rows = []
    for node in nodes:
        (kind,) = checked(cuda.cudaGraphNodeGetType(node))
        if kind == cuda.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            (params,) = checked(cuda.cudaGraphMemcpyNodeGetParams(node))
            if params.kind == cuda.cudaMemcpyKind.cudaMemcpyHostToDevice:
                width = params.extent.width
                pointer = int(params.srcPtr.ptr)
                row = dict(
                    node=hex(int(node)),
                    source=hex(pointer),
                    destination=hex(int(params.dstPtr.ptr)),
                    bytes=width,
                )
                if width <= 16:
                    payload = ct.string_at(pointer, width)
                    row["payload_hex"] = payload.hex()
                    if width == 8:
                        row["int32_pair"] = list(struct.unpack("ii", payload))
                        row["double"] = struct.unpack("d", payload)[0]
                        row["int64"] = struct.unpack("q", payload)[0]
                rows.append(row)
    return rows


def copies_from_dot(path):
    """Inspect transfers in conditional graphs without runtime node reflection.

    CUDA runtime node-type introspection can reject conditional graph nodes;
    the driver-generated DOT already supplies source pointers and extents.
    Read only live HtoD source bytes while the owning inverse is retained.
    """
    rows = []
    for block in Path(path).read_text().split('"];'):
        if "{kind | HtoD" not in block:
            continue
        line = next(line for line in block.splitlines() if "{srcPtr | dstPtr}" in line)
        source, destination = re.findall(r"0x[0-9A-Fa-f]+", line)
        width = int(re.search(r"Width \| (\d+)", block).group(1))
        node_line = next(line for line in block.splitlines() if "node handle" in line)
        node = re.findall(r"0x[0-9A-Fa-f]+", node_line)[0]
        row = dict(node=node, source=source, destination=destination, bytes=width)
        if width <= 16:
            payload = ct.string_at(int(source, 16), width)
            row["payload_hex"] = payload.hex()
            if width == 8:
                row["int32_pair"] = list(struct.unpack("ii", payload))
        rows.append(row)
    return rows


def main():
    wp.init()
    folder = Path("build/cudss-graph-audit")
    folder.mkdir(parents=True, exist_ok=True)
    trace = ApiTrace()
    rows = []
    try:
        for kind in ("spd", "symmetric", "general"):
            inv = CuDSSInverse(
                matrix(sp.diags([2.0, 3.0, 4.0, 5.0])), 1, mtype=kind, refinement_steps=0
            )
            rhs = wp.ones_like(inv.x)
            inv.apply(rhs, inv.x)
            wp.synchronize()
            trace.phase = kind + ":eager"
            inv.apply(rhs, inv.x)
            trace.phase = kind + ":caller_sync"
            wp.synchronize()
            trace.phase = kind + ":capture"
            with wp.ScopedCapture() as cap:
                inv.apply(rhs, inv.x)
            trace.phase = None
            before = copies(cap.graph)
            trace.phase = kind + ":first_replay"
            wp.capture_launch(cap.graph)
            trace.phase = kind + ":caller_sync"
            wp.synchronize()
            trace.phase = kind + ":warm_replay"
            for _ in range(3):
                wp.capture_launch(cap.graph)
            trace.phase = kind + ":caller_sync"
            wp.synchronize()
            trace.phase = None
            after = copies(cap.graph)
            solution = inv.x.numpy()
            assert np.allclose(solution.ravel(), 1 / np.arange(2.0, 6.0))
            rows.append(dict(mtype=kind, before=before, after=after, solution=solution.tolist()))
            cap.graph = None
            gc.collect()
            inv.close()
    finally:
        trace.close()
    report = dict(
        cases=rows,
        trace=trace.events,
        trace_errors=trace.errors,
        api_entries_by_phase={
            phase: dict(
                Counter(e["name"] for e in trace.events if e["site"] == 0 and e["phase"] == phase)
            )
            for phase in sorted({e["phase"] for e in trace.events})
        },
    )
    (folder / "scalar-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    if trace.errors or any(e.get("status", 0) for e in trace.events):
        raise SystemExit("Trace instrumentation or CUDA API error")
    print(
        json.dumps(
            dict(cases=rows, entries=report["api_entries_by_phase"], errors=trace.errors), indent=2
        )
    )


if __name__ == "__main__":
    main()
