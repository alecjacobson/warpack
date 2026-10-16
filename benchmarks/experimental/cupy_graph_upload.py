"""CuPy-only reproducer/control for #61: no Warp import or eigensolver code.

Run as a module and redirect stdout to a JSON file. This deliberately retains
incorrect outputs to diagnose the tested CUDA stack, not to assert correctness.
"""

import json

import cupy as cp


def main():
    n = 1_000_000
    add = cp.RawKernel(
        'extern "C" __global__ void add(double* x,int n){'
        "int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(x,1.0);}",
        "add",
    )
    rows = []
    for upload in [False, True]:
        for sync in [False, True]:
            for repeat in range(3):
                stream = cp.cuda.Stream(non_blocking=False)
                with stream:
                    x = cp.zeros(1, dtype=cp.float64)

                    def apply():
                        cp.cuda.runtime.memsetAsync(x.data.ptr, 0, 8, stream.ptr)
                        add(((n + 255) // 256,), (256,), (x, cp.int32(n)))

                    for _ in range(8):
                        apply()
                    if sync:
                        stream.synchronize()
                    stream.begin_capture()
                    apply()
                    graph = stream.end_capture()
                    if upload:
                        graph.upload(stream)
                    actual = []
                    for _ in range(3):
                        graph.launch(stream)
                        stream.synchronize()
                        actual.append(float(x.get()[0]))
                    rows.append(
                        dict(
                            upload=upload,
                            synchronized_warmup=sync,
                            repeat=repeat,
                            actual=actual,
                            expected=n,
                            passed=all(v == n for v in actual),
                        )
                    )
    print(
        json.dumps(
            dict(
                cupy_version=cp.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cuda_driver=cp.cuda.runtime.driverGetVersion(),
                rows=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
