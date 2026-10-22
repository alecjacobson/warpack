"""Checksum-pinned experimental patch for #131; never changes the installed FSAI.

Stores only the lower triangle of per-row Cholesky workspace. The adaptive
support search, arithmetic, factor packing and application remain upstream.
"""

import argparse
import difflib
import gc
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.experimental.fsai_memory import snapshot
from benchmarks.upstream.adapters import matrix

BASE_SHA256 = "35c0f0cba8318fe74fa6b4b471679fe56a17c22f40331d3c89a2053bf62d9d07"


def patched_source(source, *, base_sha256=BASE_SHA256):
    if hashlib.sha256(source.encode()).hexdigest() != base_sha256:
        raise ValueError("experimental patch requires the exact pinned upstream FSAI")
    target = source.replace(
        "wp.types.matrix(shape=(width, width), dtype=dtype)",
        "wp.types.vector(length=width * (width + 1) // 2, dtype=dtype)",
    )
    target = target.replace("matrix = wp.types.vector", "packed = wp.types.vector").replace(
        "mat = wp.types.vector", "packed = wp.types.vector"
    )
    target = target.replace("chol = matrix(dtype(0))", "chol = packed(dtype(0))").replace(
        "chol = mat(dtype(0))", "chol = packed(dtype(0))"
    )
    target, count = re.subn(
        r"chol\[(\w+), (\w+)\]", lambda m: f"chol[{m[1]} * ({m[1]} + 1) // 2 + {m[2]}]", target
    )
    if count != 17:
        raise ValueError(f"unexpected Cholesky access count: {count}")
    return target


def load_candidate(directory):
    source = Path(inspect.getfile(linear.FSAI)).read_text()
    patched = patched_source(source)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "fsai_packed_candidate.py"
    file.write_text(patched)
    (directory / "fsai-packed.patch").write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(True),
                patched.splitlines(True),
                fromfile="a/warp/_src/optim/fsai.py",
                tofile="b/warp/_src/optim/fsai.py",
            )
        )
    )
    name = "warpack_experimental_fsai_packed"
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", choices=("upstream", "packed"))
    parser.add_argument("--current", action="store_true", help="qualify the isolated newer FSAI")
    args = parser.parse_args()
    if args.current and not args.gpu:
        parser.error("--current requires --gpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.current:
        from benchmarks.experimental.fsai_current import SHA256, load

        upstream, candidate = load(), load(packed=True)
        base_sha256 = SHA256
    else:
        upstream = __import__("warp._src.optim.fsai", fromlist=["FSAI"])
        candidate = load_candidate(Path("build/fsai-packed-candidate"))
        base_sha256 = BASE_SHA256
    if args.gpu:
        a = matrix(sp.diags([-np.ones(95), 2 * np.ones(96), -np.ones(95)], [-1, 0, 1]))
        module = candidate if args.gpu == "packed" else upstream
        before = snapshot(a.device)
        begin = time.perf_counter()
        factor = module.FSAI(
            a,
            max_row_size=args.width,
            max_step_size=3,
            factor_dtype=wp.float32,
            **({} if args.current else {"apply_lanes": 8}),
        )
        after = snapshot(a.device)
        seconds = time.perf_counter() - begin
        parameters = (wp.float64, args.width, 3)
        if args.current:
            parameters = (wp.float64,) + parameters
        _, kernel = module._build_kernels(*parameters)
        report = dict(
            width=args.width,
            variant=args.gpu,
            before=before,
            after=after,
            setup_seconds=seconds,
            free_drop_bytes=before["free_bytes"] - after["free_bytes"],
            live_pool_bytes=after["pool_used_bytes"] - before["pool_used_bytes"],
            kernel_properties=wp.get_cuda_kernel_properties(kernel, device=a.device),
            factor_nnz=factor.G.nnz_sync(),
            base_sha256=base_sha256,
            candidate_sha256=hashlib.sha256(
                Path(inspect.getfile(candidate.FSAI)).read_bytes()
            ).hexdigest(),
        )
        del factor
        gc.collect()
        report["after_release"] = snapshot(a.device)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(report)
        return
    # CPU values compare without claiming timing/context memory under GPU contention.
    n = 70
    with wp.ScopedDevice("cpu"):
        a = matrix(sp.diags([-np.ones(n - 1), 2.1 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]))
    kwargs = dict(
        max_row_size=args.width,
        max_step_size=3,
        kap_tolerance=0,
        factor_dtype=wp.float32,
        reuse_pattern=True,
    )
    old, new = linear.FSAI(a, **kwargs), candidate.FSAI(a, **kwargs)
    errors = []
    for scale in (1.0, 1.2):
        if scale != 1:
            a.values.assign(a.values.numpy() * scale)
            old.update()
            new.update()
        for name in ("G", "GT"):
            x, y = getattr(old, name), getattr(new, name)
            np.testing.assert_array_equal(x.offsets.numpy(), y.offsets.numpy())
            np.testing.assert_array_equal(x.columns.numpy(), y.columns.numpy())
            error = float(np.max(abs(x.values.numpy() - y.values.numpy())))
            errors.append(error)
            np.testing.assert_allclose(x.values.numpy(), y.values.numpy(), rtol=0, atol=0)
    properties = {}
    for name, module in (
        ("upstream", __import__("warp._src.optim.fsai", fromlist=["_build_kernels"])),
        ("packed", candidate),
    ):
        _, build = module._build_kernels(wp.float64, args.width, 3)
        properties[name] = wp.get_cuda_kernel_properties(build, device="cuda:0")
    report = dict(
        width=args.width,
        max_factor_error=max(errors),
        kernel_properties=properties,
        base_sha256=BASE_SHA256,
        candidate_sha256=hashlib.sha256(
            Path(inspect.getfile(candidate.FSAI)).read_bytes()
        ).hexdigest(),
        note="CPU factor/refit equality; compiler CUDA properties only, no GPU setup timing or device-free-memory measurement",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
