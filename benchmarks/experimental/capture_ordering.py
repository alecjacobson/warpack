"""Minimal diagnostic for #61; deliberately tests unsynchronized eager warmup.

This is an upstream-runtime investigation, not an eigensolver or performance
benchmark. Warpack's solver capture methods already synchronize warmup once.
"""

import argparse
import json
from pathlib import Path

import warp as wp


@wp.kernel
def accumulate(x: wp.array[wp.float64]):
    wp.atomic_add(x, 0, wp.float64(1))


@wp.kernel
def clear(x: wp.array[wp.float64]):
    x[0] = wp.float64(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--explicit-stream", action="store_true")
    p.add_argument("--cupy", action="store_true", help="Add direct CUDA memset and CuPy controls")
    p.add_argument("--cupy-upload", action="store_true", help="Explicitly upload the CuPy graph")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.n < 1 or args.repeats < 1:
        p.error("Require positive n and repeats")
    if args.cupy_upload and not args.cupy:
        p.error("--cupy-upload requires --cupy")
    wp.init()
    device = wp.get_device("cuda:0")
    stream = wp.Stream(device) if args.explicit_stream else wp.get_stream(device)
    if args.cupy:
        import cupy as cp
    rows = []
    with wp.ScopedStream(stream):
        for method in ["memset", "kernel"] + (["cuda_memset"] if args.cupy else []):
            for sync in [False, True]:
                for repeat in range(args.repeats):
                    x = wp.zeros(1, dtype=wp.float64, device=device)

                    def apply():
                        if method == "memset":
                            x.zero_()
                        elif method == "cuda_memset":
                            cp.cuda.runtime.memsetAsync(x.ptr, 0, 8, stream.cuda_stream)
                        else:
                            wp.launch(clear, 1, [x], device=device)
                        wp.launch(accumulate, args.n, [x], device=device)

                    for _ in range(8):
                        apply()
                    if sync:
                        wp.synchronize_device(device)
                    with wp.ScopedCapture(device=device) as capture:
                        apply()
                    actual = []
                    for _ in range(3):
                        wp.capture_launch(capture.graph)
                        wp.synchronize_device(device)
                        actual.append(float(x.numpy()[0]))
                    rows.append(
                        dict(
                            backend="Warp",
                            zero_method=method,
                            synchronized_warmup=sync,
                            repeat=repeat,
                            actual=actual,
                            expected=args.n,
                            passed=all(v == args.n for v in actual),
                        )
                    )
    if args.cupy:
        raw = cp.RawKernel(
            'extern "C" __global__ void add(double* x,int n){'
            "int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(x,1.0);}",
            "add",
        )
        for sync in [False, True]:
            for repeat in range(args.repeats):
                cp_stream = cp.cuda.Stream(non_blocking=False)
                with cp_stream:
                    x_cp = cp.zeros(1, dtype=cp.float64)

                    def apply_cp():
                        cp.cuda.runtime.memsetAsync(x_cp.data.ptr, 0, 8, cp_stream.ptr)
                        raw(((args.n + 255) // 256,), (256,), (x_cp, args.n))

                    for _ in range(8):
                        apply_cp()
                    if sync:
                        cp_stream.synchronize()
                    cp_stream.begin_capture()
                    apply_cp()
                    graph = cp_stream.end_capture()
                    if args.cupy_upload:
                        graph.upload(cp_stream)
                    actual = []
                    for _ in range(3):
                        graph.launch(cp_stream)
                        cp_stream.synchronize()
                        actual.append(float(x_cp.get()[0]))
                    rows.append(
                        dict(
                            backend="CuPy",
                            zero_method="cuda_memset",
                            synchronized_warmup=sync,
                            repeat=repeat,
                            actual=actual,
                            expected=args.n,
                            passed=all(v == args.n for v in actual),
                        )
                    )
    report = dict(
        args=vars(args),
        warp_version=wp.__version__,
        device=str(device),
        rows=rows,
        failures=sum(not r["passed"] for r in rows),
        scope="Diagnostic: unsynchronized warmup failures are intentionally retained; not a production correctness test or confirmed upstream root cause.",
    )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
