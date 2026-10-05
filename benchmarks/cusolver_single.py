"""cuSOLVERSp csreigvsi comparison is explicitly a ONE-eigenpair problem."""

import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import warp as wp
from compare import wp_matrix
from cupy_backends.cuda.libs import cusolver

from warpack import CuDSSInverse, SparseOperator, SymmetricEigensolver

n = 4096
a = csp.diags(
    [-cp.full(n - 1, 0.001), cp.linspace(1.0, 100.0, n), -cp.full(n - 1, 0.001)],
    [-1, 0, 1],
    format="csr",
)
x0 = cp.random.RandomState(42).normal(size=n)
x = cp.empty(n)
mu = cp.empty(1)
handle = cp.cuda.device.get_cusolver_sp_handle()


def run():
    cusolver.dcsreigvsi(
        handle,
        n,
        a.nnz,
        a._descr.descriptor,
        a.data.data.ptr,
        a.indptr.data.ptr,
        a.indices.data.ptr,
        0.99,
        x0.data.ptr,
        10000,
        1.0e-9,
        mu.data.ptr,
        x.data.ptr,
    )


rows = []
for method in ["cuSOLVERSp csreigvsi", "CuPy eigsh"]:
    times = []
    for i in range(4):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        if method.startswith("cuSOLVER"):
            run()
        else:
            mu, v = csl.eigsh(a, k=1, which="SA", tol=1.0e-9, maxiter=30000)
            x = v[:, 0]
        cp.cuda.get_current_stream().synchronize()
        if i:
            times.append(time.perf_counter() - start)
    rows.append(
        dict(
            method=method,
            n=n,
            k=1,
            seconds=float(np.median(times)),
            eigenvalue=float(mu[0]),
            residual=float(cp.linalg.norm(a @ x - mu[0] * x)),
        )
    )
    print(rows[-1], flush=True)

Path("results/cusolver_single.json").write_text(json.dumps(rows, indent=2) + "\n")

# Same matrix and shift, with separate cuDSS setup and reusable graph timings.

wa = wp_matrix(a.get())
shifted = wp_matrix((a - 0.99 * csp.eye(n, format="csr")).get())
start = time.perf_counter()
inverse = CuDSSInverse(shifted, 8)
wp.synchronize()
setup = time.perf_counter() - start
s = SymmetricEigensolver(SparseOperator(wa), 1, ncv=8, iteration_operator=inverse, tol=1.0e-11)
graph = s.capture(15, adaptive=False)
times = []
for i in range(4):
    start = time.perf_counter()
    wp.capture_launch(graph)
    wp.synchronize()
    if i:
        times.append(time.perf_counter() - start)
w = s.eigenvalues.numpy()
v = s.eigenvectors.numpy().T
row = dict(
    method="warpack + cuDSS",
    n=n,
    k=1,
    seconds=float(np.median(times)),
    setup_seconds=setup,
    eigenvalue=float(w[0]),
    residual=float(np.linalg.norm(a.get() @ v - v * w)),
)
rows.append(row)
print(row, flush=True)
Path("results/cusolver_single.json").write_text(json.dumps(rows, indent=2) + "\n")
inverse.close()
