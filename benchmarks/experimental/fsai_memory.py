"""Isolate upstream FSAI setup memory from live factor and solver storage."""

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.upstream.adapters import matrix


def snapshot(device):
    wp.synchronize_device(device)
    return dict(
        free_bytes=device.free_memory,
        pool_used_bytes=wp.get_mempool_used_mem_current(device),
        pool_high_bytes=wp.get_mempool_used_mem_high(device),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=[16, 32, 64], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(Path(inspect.getfile(linear.FSAI)).read_bytes()).hexdigest()
    if digest != "35c0f0cba8318fe74fa6b4b471679fe56a17c22f40331d3c89a2053bf62d9d07":
        parser.error(
            "historical experiment requires installer --profile v0.2.0; use fsai_packed_storage --current for the newer revision"
        )
    wp.init()
    device = wp.get_device("cuda:0")
    n = 96
    a = matrix(sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]))
    before = snapshot(device)
    factor = linear.FSAI(
        a, max_row_size=args.width, max_step_size=3, apply_lanes=8, factor_dtype=wp.float32
    )
    after = snapshot(device)
    # Introspection only: numerical construction/application use upstream FSAI.
    from warp._src.optim.fsai import _build_kernels

    _, build = _build_kernels(wp.float64, args.width, 3)
    properties = wp.get_cuda_kernel_properties(build, device=device)
    report = dict(
        n=n,
        width=args.width,
        before=before,
        after=after,
        free_drop_bytes=before["free_bytes"] - after["free_bytes"],
        live_pool_increase_bytes=after["pool_used_bytes"] - before["pool_used_bytes"],
        factor_nnz=factor.G.nnz_sync(),
        build_kernel_properties=properties,
        fsai_sha256=hashlib.sha256(Path(inspect.getfile(linear.FSAI)).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=device.name,
    )
    del factor
    gc.collect()
    report["after_factor_release"] = snapshot(device)
    report["note"] = (
        "Fresh-process standalone FSAI on a 96-row SPD tridiagonal, no eigensolver. "
        "Free memory includes context/driver and all pools; live pool counts describe only "
        "Warp's CUDA pool. local_memory_size is bytes per kernel thread. The factor is "
        "released before the last snapshot; the CUDA context remains alive."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
