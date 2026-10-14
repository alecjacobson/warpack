"""Explicit comparison adapters; none are imported by the production library."""

import ctypes
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse


class Unsupported(Exception):
    pass


def spectra_setup(p, ncv):
    if p.api_class in "EHI":
        raise Unsupported(
            "This native adapter covers real symmetric pencils; upstream harness tracks complex/general/SVD Spectra separately"
        )
    path = Path("build/application_spectra.so")
    if not path.exists():
        raise Unsupported("Compile benchmarks/applications/spectra.cpp first")
    lib = ctypes.CDLL(str(path.resolve()))
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.app_factor.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ip,
        ip,
        dp,
        ctypes.c_int,
        ip,
        ip,
        dp,
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.app_factor.restype = ctypes.c_void_p
    lib.app_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, dp, dp, dp]
    lib.app_solve.restype = ctypes.c_int
    lib.app_free.argtypes = [ctypes.c_void_p]
    a, b = p.a.tocsc(), p.b.tocsc()
    a.sort_indices()
    b.sort_indices()
    mode = {"regular": 1, "buckling": 2}.get(p.mode, 0)
    ctx = lib.app_factor(
        p.n,
        a.nnz,
        a.indptr,
        a.indices,
        a.data,
        b.nnz,
        b.indptr,
        b.indices,
        b.data,
        mode,
        float(p.target),
    )
    if not ctx:
        raise RuntimeError("Spectra factor setup failed")
    initial = np.random.default_rng(2026).normal(size=p.n)

    def solve():
        values = np.empty(p.k)
        v = np.empty(p.n * p.k)
        count = lib.app_solve(ctx, p.k, min(ncv, p.n), initial.copy(), values, v)
        if count != p.k:
            raise RuntimeError(f"Spectra converged {count}/{p.k}")
        return values, v.reshape((p.n, p.k), order="F"), None

    return solve, lambda: lib.app_free(ctx)


def cupy_setup(p, ncv, lobpcg_tol=1e-8):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    stream = cp.cuda.Stream.from_external(wp.get_stream())
    if p.api_class in "BGH":
        raise Unsupported(
            "No native CuPy generalized shift/buckling or general eigs path in this adapter; do not apply Euclidean eigsh to M^-1 K"
        )
    with stream:
        a = cs.csr_matrix(p.a)
        if p.api_class in "EI":
            initial = cp.asarray(
                np.random.default_rng(2026).normal(size=min(p.a.shape)), dtype=a.dtype
            )

            def solve():
                with stream:
                    cp.random.seed(2026)
                    if p.mode == "svd":
                        u, w, vt = cl.svds(a, k=p.k, tol=1e-11, maxiter=10000)
                        return w, u, vt.T
                    w, v = cl.eigsh(
                        a,
                        k=p.k,
                        which="SA",
                        ncv=min(ncv, p.n),
                        tol=1e-11,
                        maxiter=10000,
                        v0=initial.copy(),
                    )
                    return w, v, None

            return solve, lambda: None, "CuPy eigsh SA" if p.api_class == "E" else "CuPy svds"
        b = cs.csr_matrix(p.b)
        # Generalized LOBPCG preserves the mass metric. This is not eigsh.
        shifted = p.b if p.mode == "regular" else p.a - p.target * p.b
        inv = CuDSSInverse(matrix(shifted), 1, mtype="spd")

        def precondition(x):
            with stream:
                one = x.ndim == 1
                xx = x[:, None] if one else x
                result = cp.empty_like(xx)
                for j in range(xx.shape[1]):
                    rhs = cp.ascontiguousarray(xx[:, j]).reshape(1, p.n)
                    inv.apply(wp.from_dlpack(rhs), inv.x)
                    result[:, j] = cp.asarray(inv.x).ravel()
                return result[:, 0] if one else result

        pre = cl.LinearOperator(
            (p.n, p.n), matvec=precondition, matmat=precondition, dtype=cp.float64
        )
        initial = cp.asarray(np.random.default_rng(2026).normal(size=(p.n, p.k)))

        def solve():
            with stream:
                cp.random.seed(2026)
                w, v = cl.lobpcg(
                    a,
                    initial.copy(),
                    B=b,
                    M=pre,
                    tol=lobpcg_tol,
                    maxiter=200,
                    largest=p.mode == "regular",
                )
                return w, v, None

        return solve, inv.close, "CuPy LOBPCG + cuDSS preconditioner"
