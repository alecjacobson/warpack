"""Explicit host/reference boundaries for the complex connection comparison."""

import ctypes
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.upstream.adapters import complex_matrix
from warpack import CGInverse
from warpack.shift import ComplexShiftInverse, realify_triplets

from .connection_warp import ConnectionEigensolver


def mass(p):
    return sp.eye(p.n, dtype=complex, format="csr") if p.b is None else p.b


def initial(p, width=None):
    rng = np.random.default_rng(2026)
    shape = p.n if width is None else (p.n, width)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def warp_setup(p, ncv, backend="cudss", cg_tol=1e-13, correction_steps=1, correction_tol=0.01):
    a, b = complex_matrix(p.a, True), complex_matrix(mass(p), True)
    shifted = complex_matrix(p.a - p.target * mass(p))
    if backend == "cudss":
        inverse = ComplexShiftInverse(shifted)
    else:
        if backend not in ("cg", "cg-fsai", "cg-corrected"):
            raise ValueError("unknown inverse backend")
        size = 4 * (len(shifted.values) + p.n)
        rows = wp.zeros(size, dtype=wp.int32)
        columns = wp.zeros_like(rows)
        values = wp.zeros(size, dtype=wp.float64)
        wp.launch(
            realify_triplets,
            p.n,
            [
                shifted.offsets,
                shifted.columns,
                shifted.values,
                p.n,
                len(shifted.values),
                wp.vec2d(0.0),
                rows,
                columns,
                values,
            ],
        )
        matrix = ws.bsr_from_triplets(2 * p.n, 2 * p.n, rows, columns, values)
        preconditioner = (
            "diag"
            if backend == "cg"
            else linear.FSAI(matrix, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
        )
        if backend == "cg-corrected":
            from benchmarks.experimental.corrected_cg import CorrectedCGInverse

            inverse = CorrectedCGInverse(
                matrix,
                tol=cg_tol,
                correction_tol=correction_tol,
                correction_steps=correction_steps,
                preconditioner=preconditioner,
            )
        else:
            inverse = CGInverse(
                matrix,
                tol=cg_tol,
                atol=0.0,
                maxiter=10000,
                preconditioner=preconditioner,
                verify_residuals=True,
            )
    start = initial(p)
    start_device = wp.array(np.r_[start.real, start.imag], dtype=wp.float64)
    solver = ConnectionEigensolver(
        a, b, inverse, p.k, ncv=ncv, target=p.target, tol=1e-10, initial_vector=start_device
    )
    return solver, inverse


def metric_ritz(p, vectors):
    """Reference-only M-orthogonal basis recovery inside the returned space.

    A generalized small Ritz solve preserves the span and rejects deficient
    metric Grams. It cannot create a missing requested eigendirection.
    """
    gram_a = vectors.conj().T @ (p.a @ vectors)
    gram_m = vectors.conj().T @ (mass(p) @ vectors)
    values, rotation = la.eigh((gram_a + gram_a.conj().T) / 2, (gram_m + gram_m.conj().T) / 2)
    return values, vectors @ rotation


def scipy_setup(p, ncv, recover_metric=False):
    factor = sla.splu((p.a - p.target * mass(p)).tocsc())
    inverse = sla.LinearOperator(p.a.shape, matvec=factor.solve, dtype=complex)

    def solve():
        # Complex Hermitian eigsh dispatches to ARPACK's nonsymmetric complex engine.
        values, vectors = sla.eigsh(
            p.a,
            M=p.b,
            k=p.k,
            sigma=p.target,
            OPinv=inverse,
            ncv=ncv,
            v0=initial(p),
            tol=1e-11,
            maxiter=10000,
        )
        return metric_ritz(p, vectors) if recover_metric else (values, vectors)

    return solve, lambda: None


def spectra_setup(p, ncv):
    lib = ctypes.CDLL(str(Path("build/connection_spectra.so").resolve()))
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    zp = np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
    lib.connection_factor.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ip,
        ip,
        zp,
        ctypes.c_int,
        ip,
        ip,
        zp,
        ctypes.c_double,
    ]
    lib.connection_factor.restype = ctypes.c_void_p
    lib.connection_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, zp, dp, zp]
    lib.connection_solve.restype = ctypes.c_int
    lib.connection_free.argtypes = [ctypes.c_void_p]
    a, b = p.a.tocsc(), mass(p).tocsc()
    a.sort_indices()
    b.sort_indices()
    ctx = lib.connection_factor(
        p.n, a.nnz, a.indptr, a.indices, a.data, b.nnz, b.indptr, b.indices, b.data, p.target
    )
    if not ctx:
        raise RuntimeError("Spectra complex SPD factorization failed")

    def solve():
        w = np.empty(p.k)
        v = np.empty(p.n * p.k, dtype=complex)
        count = lib.connection_solve(ctx, p.k, ncv, initial(p), w, v)
        if count != p.k:
            raise RuntimeError(f"Spectra converged {count}/{p.k}")
        return w, v.reshape((p.n, p.k), order="F")

    return solve, lambda: lib.connection_free(ctx)


def cupy_setup(p, maxiter=300, tol=1e-9):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        a, b = cs.csr_matrix(p.a), cs.csr_matrix(mass(p))
        inverse = ComplexShiftInverse(complex_matrix(p.a - p.target * mass(p)))
        rhs = wp.empty((1, 2 * p.n), dtype=wp.float64)
        result = wp.empty_like(rhs)
        rhs_cp, result_cp = cp.asarray(rhs), cp.asarray(result)

        def precondition(x):
            with stream:
                xx = x[:, None] if x.ndim == 1 else x
                out = cp.empty_like(xx)
                for j in range(xx.shape[1]):
                    rhs_cp[0, : p.n] = xx[:, j].real
                    rhs_cp[0, p.n :] = xx[:, j].imag
                    inverse.apply(rhs, result)
                    out[:, j] = result_cp[0, : p.n] + 1j * result_cp[0, p.n :]
                return out[:, 0] if x.ndim == 1 else out

        pre = cl.LinearOperator(
            p.a.shape, matvec=precondition, matmat=precondition, dtype=cp.complex128
        )
        start = cp.asarray(initial(p, p.k))

    def solve():
        with stream:
            w, v, history = cl.lobpcg(
                a,
                start.copy(),
                B=b,
                M=pre,
                tol=tol,
                maxiter=maxiter,
                largest=False,
                retResidualNormsHistory=True,
            )
            return w, v, history

    return solve, inverse.close


def corrected_cupy_setup(p):
    from .connection_cupy_repro import correct_helpers

    correct_helpers()
    return cupy_setup(p)


def cupy_eigsh_setup(p, ncv):
    """Native complex eigsh with explicit positive diagonal mass congruence."""
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    m = mass(p)
    off_diagonal = m - sp.diags(m.diagonal(), format="csr")
    off_diagonal.eliminate_zeros()
    if off_diagonal.nnz:
        raise ValueError("Native CuPy eigsh has no coupled mass argument; use LOBPCG")
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        scale = cp.asarray(1 / np.sqrt(m.diagonal().real))
        d = cs.diags(scale, format="csr")
        a = d @ cs.csr_matrix(p.a) @ d
        start = cp.asarray(initial(p))

    def solve():
        with stream:
            w, v = cl.eigsh(
                a, k=p.k, which="SA", ncv=ncv, tol=1e-11, maxiter=10000, v0=start.copy()
            )
            return w, scale[:, None] * v

    return solve, lambda: None
