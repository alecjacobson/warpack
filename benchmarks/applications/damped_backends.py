"""Reference and Warp setup for the assembled damped first-order matrix."""

import ctypes
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.upstream.adapters import complex_matrix
from warpack import ComplexShiftInverse, ShiftInvertEigensolver
from warpack.pencil import GeneralizedShiftInvertEigensolver


class _OwnedPencil(GeneralizedShiftInvertEigensolver):
    """Application setup owns the inverse it constructs; the public API borrows it."""

    def close(self):
        super().close()
        self.inverse.close()


def initial(case, balance=1.0):
    rng = np.random.default_rng(2026)
    vector = rng.normal(size=case.a.shape[0]) + 1j * rng.normal(size=case.a.shape[0])
    vector[: len(vector) // 2] *= balance
    return vector


def coordinates(case, balance):
    if not np.isfinite(balance) or balance <= 0:
        raise ValueError("coordinate scale must be finite and positive")
    a = case.a.astype(complex)
    half = a.shape[0] // 2
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    a.data *= np.where(rows < half, balance, 1.0) / np.where(a.indices < half, balance, 1.0)
    return a


def physical(vectors, balance, values=None, kinematic=False):
    if balance == 1.0 and not kinematic:
        return vectors
    vectors = vectors.copy()
    vectors[: len(vectors) // 2] /= balance
    if kinematic:
        vectors[len(vectors) // 2 :] = vectors[: len(vectors) // 2] * values
    vectors /= np.linalg.norm(vectors, axis=0)
    return vectors


def scipy_setup(
    case, ncv, tol=1e-10, balance=1.0, polish=0, reference_refinement=0, kinematic=False
):
    matrix = coordinates(case, balance)
    mass = case.b
    shifted = (matrix - case.target * (sp.eye(matrix.shape[0]) if mass is None else mass)).tocsc()
    factor = sla.splu(shifted)

    def apply(rhs):
        result = factor.solve(rhs)
        for _ in range(reference_refinement):
            result += factor.solve(rhs - shifted @ result)
        return result

    inverse = sla.LinearOperator(matrix.shape, matvec=apply, dtype=complex)
    start = initial(case, balance)

    def solve():
        values, vectors = sla.eigs(
            matrix,
            k=case.k,
            ncv=ncv,
            sigma=case.target,
            OPinv=inverse,
            M=mass,
            v0=start.copy(),
            tol=tol,
            maxiter=1000,
        )
        for _ in range(polish):
            vectors = apply(vectors if mass is None else mass @ vectors)
            vectors /= np.linalg.norm(vectors, axis=0)
        vectors /= np.linalg.norm(vectors, axis=0)
        return values, physical(vectors, balance, values, kinematic), {"count": len(values)}

    return solve, lambda: None


def spectra_setup(
    case,
    ncv,
    tol=1e-10,
    library="build/damped_spectra.so",
    balance=1.0,
    polish=0,
    reference_refinement=0,
    kinematic=False,
):
    lib = ctypes.CDLL(str(Path(library).resolve()))
    try:
        version = lib.damped_abi_version
    except AttributeError as error:
        raise RuntimeError("Recompile damped_spectra.so for the current reference ABI") from error
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != 2:
        raise RuntimeError("Recompile damped_spectra.so for reference ABI2")
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
        ctypes.c_int,
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
    if polish:
        lib.damped_polish.argtypes = [raw, ctypes.c_int, data]
        lib.damped_polish.restype = ctypes.c_int
    matrix = coordinates(case, balance).tocsc()
    matrix.sort_indices()
    prefix = [
        matrix.shape[0],
        matrix.nnz,
        matrix.indptr.astype(np.int32),
        matrix.indices.astype(np.int32),
        matrix.data,
    ]
    suffix = [case.target.real, case.target.imag, reference_refinement]
    if case.b is None:
        context = lib.damped_factor(*prefix, *suffix)
    else:
        if not hasattr(lib, "damped_pencil_abi_version") or lib.damped_pencil_abi_version() != 1:
            raise RuntimeError("Recompile damped_spectra.so with generalized-pencil support")
        mass = case.b.astype(complex).tocsc()
        mass.sort_indices()
        lib.damped_pencil_factor.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ints,
            ints,
            data,
            ctypes.c_int,
            ints,
            ints,
            data,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.damped_pencil_factor.restype = raw
        context = lib.damped_pencil_factor(
            *prefix,
            mass.nnz,
            mass.indptr.astype(np.int32),
            mass.indices.astype(np.int32),
            mass.data,
            *suffix,
        )
    if not context:
        raise RuntimeError("Spectra/Eigen sparse factorization failed")
    start = np.ascontiguousarray(initial(case, balance))

    def solve():
        values = np.empty(case.k, complex)
        vectors = np.empty(matrix.shape[0] * case.k, complex)
        stats = np.zeros(4, np.int32)
        result = lib.damped_solve(context, case.k, ncv, start, tol, 1000, values, vectors, stats)
        if result != 0:
            raise RuntimeError(f"Spectra returned {result}, statistics {stats.tolist()}")
        for _ in range(polish):
            if lib.damped_polish(context, case.k, vectors) != 0:
                raise RuntimeError("Spectra reference inverse polishing failed")
            shaped = vectors.reshape((matrix.shape[0], case.k), order="F")
            shaped /= np.linalg.norm(shaped, axis=0)
        return (
            values,
            physical(
                vectors.reshape((matrix.shape[0], case.k), order="F"), balance, values, kinematic
            ),
            dict(
                count=int(stats[0]),
                iterations=int(stats[1]),
                operations=int(stats[2]),
                status=int(stats[3]),
            ),
        )

    return solve, lambda: lib.damped_free(context)


def warp_setup(
    case,
    ncv,
    tol=1e-10,
    backend="cudss",
    balance=1.0,
    polish=0,
    kinematic=False,
    inner_restart=None,
):
    start = initial(case, balance)
    device_start = wp.array(np.stack([start.real, start.imag], axis=-1), dtype=wp.vec2d)
    if case.b is not None:
        if balance != 1.0 or polish or kinematic:
            raise ValueError("generalized pencil benchmark currently uses original coordinates")
        inverse = ComplexShiftInverse(
            complex_matrix(case.a - case.target * case.b),
            backend=backend,
            inner_tol=1e-14,
            inner_maxiter=1000,
            inner_restart=inner_restart,
        )
        try:
            return _OwnedPencil(
                complex_matrix(case.a),
                complex_matrix(case.b),
                inverse,
                case.k,
                sigma=case.target,
                ncv=ncv,
                tol=tol,
                initial_vector=device_start,
            )
        except Exception:
            inverse.close()
            raise
    cls = ShiftInvertEigensolver
    operators = [complex_matrix(case.a)]
    kwargs = {}
    if balance != 1.0 or polish or kinematic:
        from benchmarks.experimental.damped_coordinates import DampedCoordinates

        cls = DampedCoordinates
        operators.append(complex_matrix(coordinates(case, balance)))
        kwargs["coordinate_scale"] = balance
        kwargs["polish_steps"] = polish
        kwargs["kinematic"] = kinematic
    return cls(
        *operators,
        case.k,
        case.target,
        ncv=ncv,
        tol=tol,
        backend=backend,
        inner_tol=1e-13,
        inner_maxiter=1000,
        initial_vector=device_start,
        inner_restart=inner_restart,
        **kwargs,
    )
