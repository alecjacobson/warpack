"""Accuracy-gated Anvil sparse-product microbenchmark; not solver timings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.experimental.compare_block import prepare
from benchmarks.experimental.sparse_layout import product


def host_matrix(a):
    a.nnz_sync()
    data = a.values.numpy()[: a.nnz].astype(np.float64)
    constructor = sp.csr_matrix if a.block_shape == (1, 1) else sp.bsr_matrix
    return constructor((data, a.columns.numpy()[: a.nnz], a.offsets.numpy()), shape=a.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    before = other_gpu_processes()
    _, _, shifted, _, _, _, _, _ = prepare(argparse.Namespace(case="anvil", shift=10000))
    shifted.nnz_sync()
    pre = linear.FSAI(shifted, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
    a = ws.bsr_copy(shifted, scalar_type=wp.float32)
    source = np.random.default_rng(42).normal(size=(24, a.shape[0])).astype(np.float32)
    x = wp.array(source, dtype=wp.float32)
    ax, gx, gtx = [wp.empty_like(x) for _ in range(3)]
    # Independent FP64 reference using the actual FP32 matrix/factor coefficients.
    reference_a = (host_matrix(a) @ source.astype(np.float64).T).T
    reference_pre = (host_matrix(pre.GT) @ (host_matrix(pre.G) @ source.astype(np.float64).T)).T
    rows = []
    for lanes in (8, 4, 2, 1, 16, 32):

        def apply():
            product(a, x, ax, lanes)
            product(pre.G, x, gx, lanes)
            product(pre.GT, gx, gtx, lanes)

        apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            apply()
        wp.capture_launch(capture.graph)
        wp.synchronize()
        times = [timed(lambda: wp.capture_launch(capture.graph))[0] for _ in range(5)]
        errors = [
            float(
                np.linalg.norm(result.numpy().astype(float) - reference) / np.linalg.norm(reference)
            )
            for result, reference in ((ax, reference_a), (gtx, reference_pre))
        ]
        rows.append(
            dict(
                lanes_per_row=lanes,
                rhs_per_warp=32 // lanes,
                samples_seconds=times,
                median_seconds=float(np.median(times)),
                relative_frobenius_errors=errors,
                accepted=max(errors) < 5e-6,
            )
        )
        print(rows[-1], flush=True)
    after = other_gpu_processes()
    Path(args.out).write_text(
        json.dumps(
            dict(
                rows=rows,
                width=24,
                n=a.shape[0],
                precision="FP32 sparse products; FP64 independent reference",
                scope="One shifted stiffness product and upstream FSAI G/GT application. Not CG or whole eigensolver timing; no replacement factorization.",
                other_gpu_processes_before=before,
                other_gpu_processes_after=after,
                timing_eligible=not (before or after),
                source_sha256={
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path("benchmarks/experimental/sparse_layout.py"))
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
