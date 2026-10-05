"""Full dragon comparison. Assembly and CPU transfers are outside solve timing."""

import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from warpack import CuDSSInverse
from warpack.fem import assemble_rest, mass_normalized, read_mesh


def main():
    x, t = read_mesh("../bbw-comparison/dragon-H/dragon.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, m, mass = assemble_rest(x, t)
    h.nnz_sync()
    a = mass_normalized(h, mass, shift=1.0)
    a.nnz_sync()
    n = a.shape[0]
    results = []
    start = time.perf_counter()
    inverse = CuDSSInverse(a, 1)
    wp.synchronize()
    factor = time.perf_counter() - start
    inverse.b.zero_()
    inverse.apply(inverse.b, inverse.x)
    wp.synchronize()
    # Sharing device buffers; all matvecs run on the same stream as cuDSS.
    stream = cp.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream)
    scalar = inverse.solver._csr
    from cupyx.scipy.sparse import csr_matrix

    with stream:
        ca = csr_matrix(
            (
                cp.asarray(scalar.values),
                cp.asarray(scalar.columns),
                cp.asarray(scalar.row_offsets),
            ),
            shape=(n, n),
        )

        def matvec(v):
            src = wp.from_dlpack(v.reshape(1, n))
            inverse.apply(src, inverse.x)
            return cp.asarray(inverse.x).reshape(n).copy()

        op = csl.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        start = time.perf_counter()
        w, v = csl.eigsh(op, k=26, ncv=64, which="LA", tol=1.0e-12, maxiter=2000)
        stream.synchronize()
        elapsed = time.perf_counter() - start
        vals = 1 / w - 1
        residual = cp.linalg.norm(ca @ v - v * (vals + 1), axis=0) / cp.maximum(1, cp.abs(vals))
        results.append(
            dict(
                method="CuPy eigsh + cuDSS inverse",
                factor_seconds=factor,
                solve_seconds=elapsed,
                relative_residual=float(residual.max()),
                eigenvalues=sorted(cp.asnumpy(vals).tolist()),
            )
        )
        print(results[-1], flush=True)
    # CPU ARPACK uses the same shifted, normalized CSR and reports factor time.
    sa = sp.csr_matrix(
        (scalar.values.numpy(), scalar.columns.numpy(), scalar.row_offsets.numpy()),
        shape=(n, n),
    )
    Path("results/dragon_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    start = time.perf_counter()
    lu = sla.splu(sa.tocsc())
    factor = time.perf_counter() - start
    op = sla.LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)
    start = time.perf_counter()
    w, v = sla.eigsh(op, k=26, ncv=64, which="LA", tol=1.0e-12)
    elapsed = time.perf_counter() - start
    vals = 1 / w - 1
    res = np.linalg.norm(sa @ v - v * (vals + 1), axis=0) / np.maximum(1, np.abs(vals))
    results.append(
        dict(
            method="SciPy ARPACK + SuperLU inverse",
            factor_seconds=factor,
            solve_seconds=elapsed,
            relative_residual=float(res.max()),
            eigenvalues=sorted(vals.tolist()),
        )
    )
    print(results[-1], flush=True)
    Path("results/dragon_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    inverse.close()


if __name__ == "__main__":
    main()
