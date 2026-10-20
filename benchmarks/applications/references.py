"""Explicit comparison adapters; none are imported by the production library."""

import ctypes
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse


class Unsupported(Exception):
    pass


def spectra_setup(p, ncv):
    if p.mode == "svd":
        return spectra_svd_setup(p, ncv)
    if p.api_class in "EH":
        raise Unsupported(
            "This native adapter covers real symmetric pencils; upstream harness tracks complex/general Spectra separately"
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
    mode = {"regular": 1, "buckling": 2, "cayley": 3}.get(p.mode, 0)
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


def spectra_svd_setup(p, ncv):
    path = Path("build/application_spectra.so")
    if not path.exists():
        raise Unsupported("Compile benchmarks/applications/spectra.cpp first")
    lib = ctypes.CDLL(str(path.resolve()))
    if not hasattr(lib, "app_svd_setup"):
        raise Unsupported("Recompile application_spectra.so for the SVD adapter")
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.app_svd_setup.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ip, ip, dp]
    lib.app_svd_setup.restype = ctypes.c_void_p
    lib.app_svd_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, dp, dp, dp]
    lib.app_svd_solve.restype = ctypes.c_int
    lib.app_svd_free.argtypes = [ctypes.c_void_p]
    a = p.a.tocsc()
    a.sort_indices()
    context = lib.app_svd_setup(*a.shape, a.nnz, a.indptr, a.indices, a.data)
    if not context:
        raise RuntimeError("Spectra SVD setup failed")

    def solve():
        values = np.empty(p.k)
        u, v = np.empty(a.shape[0] * p.k), np.empty(a.shape[1] * p.k)
        count = lib.app_svd_solve(context, p.k, min(ncv, min(a.shape)), values, u, v)
        if count != p.k:
            raise RuntimeError(f"Spectra SVD converged {count}/{p.k}")
        return (
            values,
            u.reshape((a.shape[0], p.k), order="F"),
            v.reshape((a.shape[1], p.k), order="F"),
        )

    return solve, lambda: lib.app_svd_free(context)


def cupy_setup(
    p,
    ncv,
    lobpcg_tol=1e-8,
    standard=False,
    batched=False,
    standard_which="SA",
    eigsh_tol=1e-11,
    lobpcg_maxiter=200,
):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    if batched and (standard or p.api_class not in "ACDF"):
        raise Unsupported("Batched cuDSS comparison requires a generalized CuPy LOBPCG path")
    mass_diagonal = None
    if standard:
        if standard_which not in ("SA", "LA", "LM"):
            raise Unsupported(
                "Native CuPy eigsh supports SA/LA/LM, not SM or a shift-invert argument"
            )
        if p.api_class not in "ACD" or p.b is None:
            raise Unsupported(
                "Explicit CuPy eigsh comparison requires diagonal-mass FEM or Hodge data"
            )
        diagonal = p.b.diagonal()
        delta = p.b - sp.diags(diagonal, format="csr")
        delta.eliminate_zeros()
        if delta.nnz or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
            raise Unsupported(
                "CuPy eigsh has no generalized mass argument; this path requires positive diagonal mass, otherwise use LOBPCG"
            )
        if not np.all(diagonal == 1):
            mass_diagonal = diagonal
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    if p.api_class in "BGH":
        raise Unsupported(
            "No native CuPy generalized shift/buckling or general eigs path in this adapter; do not apply Euclidean eigsh to M^-1 K"
        )
    with stream:
        a = cs.csr_matrix(p.a)
        if standard:
            scale = None if mass_diagonal is None else 1 / cp.sqrt(cp.asarray(mass_diagonal))
            if scale is not None:
                transform = cs.diags(scale, format="csr")
                a = transform @ a @ transform
            effective_ncv = min(max(ncv, p.k + 2), p.n - 1)
            initial = cp.asarray(np.random.default_rng(2026).normal(size=p.n))

            def solve():
                with stream:
                    cp.random.seed(2026)
                    w, v = cl.eigsh(
                        a,
                        k=p.k,
                        which=standard_which,
                        ncv=effective_ncv,
                        tol=eigsh_tol,
                        maxiter=10000,
                        v0=initial.copy(),
                    )
                    return w, v if scale is None else scale[:, None] * v, None

            mass_label = "identity mass" if scale is None else "diagonal mass scaling"
            return (
                solve,
                lambda: None,
                f"CuPy eigsh {standard_which}, {mass_label}, ncv={effective_ncv}",
            )
        if p.api_class in "EI":
            initial = cp.asarray(
                np.random.default_rng(2026).normal(size=min(p.a.shape)), dtype=a.dtype
            )

            def solve():
                with stream:
                    cp.random.seed(2026)
                    if p.mode == "svd":
                        u, w, vt = cl.svds(a, k=p.k, ncv=ncv, tol=1e-11, maxiter=10000)
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
        if p.n < 5 * p.k:
            raise Unsupported(
                "CuPy LOBPCG uses a dense fallback below n=5*k; excluded from this sparse comparison"
            )
        b = cs.csr_matrix(p.b)
        # Generalized LOBPCG preserves the mass metric. This is not eigsh.
        shifted = p.b if p.mode == "regular" else p.a - p.target * p.b
        inv = CuDSSInverse(matrix(shifted), p.k if batched else 1, mtype="spd")
        rhs_block, solution_block = cp.asarray(inv.b), cp.asarray(inv.x)

        def precondition(x):
            with stream:
                one = x.ndim == 1
                xx = x[:, None] if one else x
                if batched:
                    count = xx.shape[1]
                    if count > p.k:
                        raise ValueError("LOBPCG preconditioner block exceeds setup capacity")
                    inv.b.zero_()
                    rhs_block[:count] = xx.T
                    inv.apply(inv.b, inv.x)
                    result = solution_block[:count].T.copy()
                    return result[:, 0] if one else result
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
                    maxiter=lobpcg_maxiter,
                    largest=p.mode == "regular",
                )
                return w, v, None

        label = "CuPy LOBPCG + cuDSS preconditioner"
        if batched:
            label = f"CuPy LOBPCG + batched cuDSS preconditioner, padded width={p.k}"
        return solve, inv.close, label
