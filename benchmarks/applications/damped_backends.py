"""Reference and Warp setup for the assembled damped first-order matrix."""

import ctypes
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.upstream.adapters import complex_matrix
from warpack import ShiftInvertEigensolver


def initial(case):
    rng = np.random.default_rng(2026)
    return rng.normal(size=case.a.shape[0]) + 1j * rng.normal(size=case.a.shape[0])


def scipy_setup(case, ncv, tol=1e-10):
    matrix = case.a.astype(complex)
    factor = sla.splu((matrix - case.target * sp.eye(matrix.shape[0])).tocsc())
    inverse = sla.LinearOperator(matrix.shape, matvec=factor.solve, dtype=complex)
    start = initial(case)

    def solve():
        values, vectors = sla.eigs(
            matrix,
            k=case.k,
            ncv=ncv,
            sigma=case.target,
            OPinv=inverse,
            v0=start.copy(),
            tol=tol,
            maxiter=1000,
        )
        return values, vectors, {"count": len(values)}

    return solve, lambda: None


def spectra_setup(case, ncv, tol=1e-10, library="build/damped_spectra.so"):
    lib = ctypes.CDLL(str(Path(library).resolve()))
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    data = np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
    raw = ctypes.c_void_p
    lib.damped_factor.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ints,
        ints,
        data,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.damped_factor.restype = raw
    lib.damped_solve.argtypes = [
        raw,
        ctypes.c_int,
        ctypes.c_int,
        data,
        ctypes.c_double,
        ctypes.c_int,
        data,
        data,
        ints,
    ]
    lib.damped_solve.restype = ctypes.c_int
    lib.damped_free.argtypes = [raw]
    matrix = case.a.astype(complex).tocsc()
    matrix.sort_indices()
    context = lib.damped_factor(
        matrix.shape[0],
        matrix.nnz,
        matrix.indptr.astype(np.int32),
        matrix.indices.astype(np.int32),
        matrix.data,
        case.target.real,
        case.target.imag,
    )
    if not context:
        raise RuntimeError("Spectra/Eigen sparse factorization failed")
    start = np.ascontiguousarray(initial(case))

    def solve():
        values = np.empty(case.k, complex)
        vectors = np.empty(matrix.shape[0] * case.k, complex)
        stats = np.zeros(4, np.int32)
        result = lib.damped_solve(context, case.k, ncv, start, tol, 1000, values, vectors, stats)
        if result != 0:
            raise RuntimeError(f"Spectra returned {result}, statistics {stats.tolist()}")
        return (
            values,
            vectors.reshape((matrix.shape[0], case.k), order="F"),
            dict(
                count=int(stats[0]),
                iterations=int(stats[1]),
                operations=int(stats[2]),
                status=int(stats[3]),
            ),
        )

    return solve, lambda: lib.damped_free(context)


def warp_setup(case, ncv, tol=1e-10, backend="cudss"):
    start = initial(case)
    device_start = wp.array(np.stack([start.real, start.imag], axis=-1), dtype=wp.vec2d)
    return ShiftInvertEigensolver(
        complex_matrix(case.a),
        case.k,
        case.target,
        ncv=ncv,
        tol=tol,
        backend=backend,
        inner_tol=1e-13,
        inner_maxiter=1000,
        initial_vector=device_start,
    )
