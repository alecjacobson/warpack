"""Paired FP64 product microbenchmarks, separate from eigensolver acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.experimental.block_kernels import rotate_kernel
from warpack import kernels as K
from warpack import linalg_tiles as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=283857)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    before = other_gpu_processes()
    x = wp.empty((72, args.n), dtype=wp.float64)
    coeff = wp.empty((24, 72), dtype=wp.float64)
    out = wp.empty((24, args.n), dtype=wp.float64)
    wp.launch(K.random_block, x.shape, [x, 19])
    wp.launch(K.random_block, coeff.shape, [coeff, 23])
    rows = []
    reference = None
    for m, n, k in [(4, 32, 8), (8, 64, 8), (16, 64, 8), (16, 128, 8), (8, 128, 16)]:
        kernel = T.rotate_tiled if (m, n, k) == (4, 32, 8) else rotate_kernel(m, n, k)

        def apply():
            wp.launch_tiled(
                kernel,
                dim=((24 + m - 1) // m, (args.n + n - 1) // n),
                inputs=[x, coeff, out],
                block_dim=128,
            )

        apply()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            apply()
        times = [timed(lambda: wp.capture_launch(capture.graph))[0] for _ in range(5)]
        result = out.numpy()
        if reference is None:
            reference = result
        error = float(np.max(abs(result - reference)))
        rows.append(
            dict(
                tile=[m, n, k],
                seconds=times,
                median_seconds=float(np.median(times)),
                max_difference_from_baseline=error,
            )
        )
        print(rows[-1], flush=True)
    after = other_gpu_processes()
    Path(args.out).write_text(
        json.dumps(
            dict(
                n=args.n,
                rows=rows,
                other_gpu_processes_before=before,
                other_gpu_processes_after=after,
                timing_eligible=not (before or after),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
