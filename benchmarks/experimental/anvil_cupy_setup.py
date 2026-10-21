"""Measure independent CuPy Anvil setup without constructing a Warp eigensolver."""

import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np
import warp as wp
import warp.optim.linear as linear

from benchmarks.applications.compare import timed
from benchmarks.experimental.compare_block import cupy_trials, prepare
from warpack import BatchedCGPreconditioner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    output = parser.parse_args().out
    profile_path = Path("results/release_v0_2_0/release-public.json")
    config = json.loads(profile_path.read_text())["args"]
    config["repeats"] = 1  # One validation trial after warmup; not a new solve comparison.
    args = argparse.Namespace(**config)
    wp.init()
    a, _, shifted, constraints, sa, sb, k, check = prepare(args)
    shifted.nnz_sync()
    initial = np.random.default_rng(42).normal(size=(a.shape[0], args.width))

    def construct():
        pre = linear.FSAI(
            shifted,
            max_row_size=args.fsai_width,
            max_step_size=3,
            apply_lanes=8,
            factor_dtype=wp.float32,
        )
        cg = BatchedCGPreconditioner(
            shifted,
            args.width,
            tol=args.inner_tol,
            atol=0.0,
            maxiter=args.inner_cg,
            preconditioner=pre,
            working_dtype=wp.float32,
            fsai_layout=args.fsai_layout,
            matrix_layout=args.matrix_layout,
        )
        return pre, cg

    construction_seconds, (pre, cg) = timed(construct)
    rows, adapter = cupy_trials(sa, sb, pre, initial, constraints, args, k, check, cg=cg)
    accepted = all(
        row["quality"]["accepted"] and row["full_block_absolute_tolerance_met"] for row in rows
    )
    report = dict(
        accepted=bool(accepted),
        args=config,
        n=a.shape[0],
        k=k,
        setup_seconds=construction_seconds + adapter["reference_setup_seconds"],
        factor_and_base_cg_setup_seconds=construction_seconds,
        cupy_setup=adapter,
        setup_scope=(
            "Independent FSAI construction, base CG and all active-width workspaces, "
            "FP32 matrix copy, CuPy matrix/start/constraint upload, LinearOperator, "
            "and inner adapter capture. Includes encountered JIT with existing disk caches. "
            "Excludes imports/runtime initialization, mesh assembly and mass normalization, "
            "reference loading, host random-start generation, and outer eigensolve warmup. "
            "No factor, CG state, or graph reused from a Warp eigensolve."
        ),
        validation_rows=rows,
        gpu=wp.get_device().name,
        versions=dict(warp=wp.__version__, cupy=cp.__version__),
        sources_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).relative_to(Path.cwd()),
                Path("benchmarks/experimental/compare_block.py"),
                Path("benchmarks/experimental/batched_cg.py"),
                profile_path,
                Path("anvil for export-ftetwild.mesh"),
            )
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Independent setup seconds:", report["setup_seconds"], "accepted:", accepted)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
