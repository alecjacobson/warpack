"""Matched Anvil FSAI application timings, including layout conversion costs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear

from benchmarks.applications.compare import timed
from benchmarks.experimental.batched_cg import RepeatedPreconditioner
from benchmarks.graph_audit import graph_summary
from examples.problems.elasticity import assemble_rest, mass_normalized, read_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    x, t = read_mesh("anvil for export-ftetwild.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass, shift=100)
    a.nnz_sync()
    pre = linear.FSAI(a, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
    wp.synchronize()
    factors = []
    for factor in (pre.G, pre.GT):
        factor.nnz_sync()
        factors.append(
            sp.csr_matrix(
                (
                    factor.values.numpy()[: factor.nnz],
                    factor.columns.numpy()[: factor.nnz],
                    factor.offsets.numpy(),
                ),
                shape=factor.shape,
            )
        )
    report = dict(
        n=a.shape[0],
        shift=100,
        fsai_width=32,
        factor_dtype="float32",
        working_dtype="float32",
        warp_runtime=wp.__version__,
        rows=[],
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("benchmarks/experimental/coalesced_fsai.py"),
                Path("benchmarks/experimental/batched_cg.py"),
                Path("benchmarks/experimental/block_kernels.py"),
            ]
        },
        timing_scope="Warm graph replay plus synchronization, one application of G then GT including all layout conversions and output update. Alternating order, independent SciPy FP64 accumulation check after every sample; no host work inside timed graph.",
    )
    for width in (4, 8, 16, 24):
        host = np.random.default_rng(42 + width).normal(size=(width, a.shape[0])).astype(np.float32)
        expected = (factors[1].astype(np.float64) @ (factors[0].astype(np.float64) @ host.T)).T
        norm = np.linalg.norm(expected)
        source = wp.array(host.ravel(), dtype=wp.float32)
        offsets = wp.array(np.arange(width + 1, dtype=np.int32) * a.shape[0], dtype=wp.int32)
        cases = []
        for layout in ("rows", "columns"):
            adapter = RepeatedPreconditioner(
                pre, width, a.shape[0], offsets, a.device, dtype=wp.float32, layout=layout
            )
            out = wp.empty_like(source)
            adapter.matvec(source, out, out, 1.0, 0.0)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                adapter.matvec(source, out, out, 1.0, 0.0)
            for _ in range(5):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            dot = args.out.with_suffix(f".{width}.{layout}.dot")
            wp.capture_debug_dot_print(capture.graph, str(dot))
            audit = graph_summary(dot)
            dot.unlink()
            cases.append((layout, adapter, out, capture.graph, audit))
        for trial in range(args.repeats):
            for layout, adapter, out, graph, audit in cases[:: 1 if trial % 2 == 0 else -1]:
                seconds, _ = timed(lambda: wp.capture_launch(graph))
                actual = out.numpy().reshape(host.shape)
                relative = float(np.linalg.norm(actual - expected) / norm)
                accepted = bool(np.isfinite(actual).all() and relative < 2e-6)
                report["rows"].append(
                    dict(
                        width=width,
                        layout=layout,
                        trial=trial,
                        seconds=seconds,
                        relative_error=relative,
                        accepted=accepted,
                        graph=audit,
                    )
                )
        print(
            width,
            {
                layout: float(
                    np.median(
                        [
                            r["seconds"]
                            for r in report["rows"]
                            if r["width"] == width and r["layout"] == layout
                        ]
                    )
                )
                for layout in ("rows", "columns")
            },
            flush=True,
        )
    report["accepted"] = all(row["accepted"] for row in report["rows"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("FSAI application accuracy rejected")


if __name__ == "__main__":
    main()
