"""Instrument FSAI construction on one shared Anvil matrix; no eigensolve timings."""

import argparse
import gc
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.sparse as sp
from warp.optim import linear

from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.fsai_block_access import load as load_block_access
from benchmarks.experimental.fsai_current import load as load_current
from examples.problems.elasticity import assemble_rest, mass_normalized, read_mesh


def identity(factor):
    h = hashlib.sha256()
    g = factor.G
    count = g.nnz_sync()
    for data in (g.offsets.numpy(), g.columns.numpy()[:count], g.values.numpy()[:count]):
        h.update(data.tobytes())
    return dict(nnz=count, sha256=h.hexdigest(), truncated_rows=factor.truncated_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vertices, tets = read_mesh("anvil for export-ftetwild.mesh")
    vertices = (vertices - vertices.mean(0)) / np.ptp(vertices, axis=0).max()
    hessian, _, mass = assemble_rest(vertices, tets)
    a = mass_normalized(hessian, mass, shift=100.0)
    a.nnz_sync()
    begin = time.perf_counter()
    scalar = sp.bsr_copy(a, block_shape=(1, 1))
    scalar.nnz_sync()
    wp.synchronize()
    conversion = time.perf_counter() - begin
    candidates = [
        ("installed-block", linear.FSAI, a),
        ("current-block", load_current().FSAI, a),
        ("packed-block", load_current(packed=True).FSAI, a),
        ("packed-scalar", load_current(packed=True).FSAI, scalar),
        ("flat-block", load_block_access().FSAI, a),
        ("view-block", load_block_access("view").FSAI, a),
    ]
    rows = []
    expected = None
    for name, constructor, matrix in candidates:
        for trial in range(args.repeats + 1):
            wp.synchronize()
            begin = time.perf_counter()
            with wp.ScopedTimer("FSAI setup", cuda_filter=wp.TIMING_ALL, print=False) as timer:
                factor = constructor(
                    matrix, max_row_size=32, max_step_size=3, factor_dtype=wp.float32
                )
            wp.synchronize()
            seconds = time.perf_counter() - begin
            phases = {}
            for item in timer.timing_results:
                phases[item.name] = phases.get(item.name, 0.0) + item.elapsed
            result = identity(factor)
            if expected is None:
                expected = result
            row = dict(
                implementation=name,
                warmup=trial == 0,
                seconds=seconds,
                cuda_ms=phases,
                factor=result,
                identical_to_installed=result == expected,
                source_sha256=hashlib.sha256(
                    Path(inspect.getfile(constructor)).read_bytes()
                ).hexdigest(),
            )
            rows.append(row)
            print(name, trial, seconds, result == expected, flush=True)
            del factor
            gc.collect()
    report = dict(
        n=a.shape[0],
        repeats=args.repeats,
        scalar_conversion_seconds=conversion,
        dependency=warp_dependencies(),
        rows=rows,
        instrumented=True,
        timing_scope="Warm construction on a shared assembled matrix, with CUDA activity instrumentation. Warmup includes compilation. Scalar conversion is separately charged for packed-scalar. This is not an end-to-end eigensolve comparison.",
        all_factors_identical=all(r["identical_to_installed"] for r in rows),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
