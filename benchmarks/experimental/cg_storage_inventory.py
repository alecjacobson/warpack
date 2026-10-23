"""Inspect active-CG named scratch allocations on a shape-matched diagonal fixture.

This is an allocation diagnostic, not an eigensolve/performance comparison.
Matrix/factor storage and driver/runtime allocations are outside named counts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
import warp.optim.linear as linear
import warp.sparse as sparse

from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.source_provenance import source_manifest
from warpack import BatchedCGPreconditioner
from warpack._active_cg import DeviceBucketedCG


def summarize(arrays):
    intervals = []
    for array in arrays:
        if array is None or array.size == 0:
            continue
        # Span includes any stride gaps; current counted arrays are contiguous
        # or prefix views. Merge overlapping views rather than counting twice.
        span = wp.types.type_size_in_bytes(array.dtype)
        span += sum((n - 1) * stride for n, stride in zip(array.shape, array.strides))
        intervals.append((array.ptr, array.ptr + span))
    merged = []
    for start, end in sorted(intervals):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return dict(
        array_views=len(intervals),
        nonoverlapping_spans=len(merged),
        named_bytes=sum(b - a for a, b in merged),
    )


def inventory(owner):
    groups = {
        "outer_packing": [*owner.rhs, *owner.outputs],
        "outer_source": [owner.source],
        "outer_control": [
            owner.rows,
            owner.count,
            owner.widths,
            owner.conditions,
            owner.stats,
            owner.usage,
        ],
        "adapter_b_x": [],
        "matrix_application": [],
        "factor_application": [],
        "adapter_control": [],
        "upstream_cg_vectors": [],
        "upstream_cg_reductions_control": [],
    }
    for adapter in owner.states:
        groups["adapter_b_x"].extend([adapter.b, adapter.x])
        groups["adapter_control"].extend([adapter.offsets, adapter.stats])
        for key, operation in [
            ("matrix_application", adapter.operator),
            ("factor_application", adapter.preconditioner),
        ]:
            if operation is None:
                continue
            if operation.coalesced is not None:
                groups[key].extend([operation.coalesced.first, operation.coalesced.second])
            else:
                groups[key].extend(
                    [getattr(operation, "tmp", None), getattr(operation, "out", None)]
                )
        # Read-only diagnostic of the pinned upstream state. No field is replaced.
        state = adapter.state
        groups["upstream_cg_vectors"].extend([state._r_and_z_buf, state._p_and_Ap])
        groups["upstream_cg_reductions_control"].extend(
            [state._residuals, state._cur_iter_and_condition]
        )
        groups["upstream_cg_reductions_control"].extend(
            x for x in vars(state._tiled_dot).values() if isinstance(x, wp.array)
        )
    return dict(
        categories={name: summarize(arrays) for name, arrays in groups.items()},
        all_named=summarize([a for arrays in groups.values() for a in arrays]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=34560)
    parser.add_argument("--width", type=int, default=204)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 3 or args.n % 3 or args.width < 1:
        parser.error("positive width and n divisible by3 required")
    if args.out.exists():
        parser.error("refusing overwrite")
    indices = wp.array(np.arange(args.n // 3, dtype=np.int32), dtype=wp.int32)
    blocks = np.tile(np.eye(3), (args.n // 3, 1, 1))
    matrix = sparse.bsr_from_triplets(
        args.n // 3, args.n // 3, indices, indices, wp.array(blocks, dtype=wp.mat33d)
    )
    pre = linear.FSAI(matrix, max_row_size=8, factor_dtype=wp.float32)
    base = BatchedCGPreconditioner(
        matrix,
        args.width,
        working_dtype=wp.float32,
        preconditioner=pre,
        matrix_layout="columns",
        fsai_layout="columns",
        maxiter=25,
        tol=0.01,
    )
    owner = DeviceBucketedCG(base, wp.zeros(args.width, dtype=wp.float64), 1e-11, bucket_size=4)
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        inventory=inventory(owner),
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(type(pre)),
        scope="Actual named scratch spans on a diagonal3x3 block fixture with the specified n/width and FP32 column layouts. All counts use device metadata without host value reads; overlapping views are deduplicated within each category and across the total. Excludes sparse matrix/FSAI factors, outer eigensolver storage and runtime/pools. Not peak memory, complete FEM memory or timing evidence. Private upstream fields are read only for this diagnostic.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["inventory"], indent=2))


if __name__ == "__main__":
    main()
