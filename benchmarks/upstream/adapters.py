"""Test-only NumPy/CuPy boundary adapters; all solver arithmetic runs in Warp."""

import time

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws

from warpack import (
    ComplexCSR,
    CuDSSInverse,
    EigenpairEvaluation,
    HermitianCSR,
    HermitianEigensolver,
    KrylovSchur,
    PartialSVD,
    ShiftInvertEigensolver,
    SparseOperator,
)


def matrix(a):
    a = sp.csr_matrix(a)
    a.sum_duplicates()
    a.eliminate_zeros()
    coo = a.tocoo()
    return ws.bsr_from_triplets(
        *a.shape,
        wp.array(coo.row, dtype=wp.int32),
        wp.array(coo.col, dtype=wp.int32),
        wp.array(coo.data, dtype=wp.float64),
    )


def complex_matrix(a, hermitian=False):
    a = sp.csr_matrix(a, dtype=np.complex128)
    a.sum_duplicates()
    return (HermitianCSR if hermitian else ComplexCSR)(
        a.shape[0],
        wp.array(a.indptr, dtype=wp.int32),
        wp.array(a.indices, dtype=wp.int32),
        wp.array(np.column_stack([a.data.real, a.data.imag]), dtype=wp.vec2d),
    )


def host(a):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator

    if isinstance(a, LinearOperator):
        # Only test-fixture conversion. Never a production matvec callback.
        a = a.matmat(cp.eye(a.shape[1], dtype=a.dtype))
    return a.get() if hasattr(a, "get") else a


def output(a):
    x = a.numpy()
    return x[..., 0] + 1j * x[..., 1] if a.dtype == wp.vec2d else x


def execute(solver, iterations=300, repeats=1, **capture_options):
    start = time.perf_counter()
    graph = solver.capture(iterations, **capture_options)
    wp.capture_launch(graph)
    wp.synchronize()
    warmup = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        samples.append(time.perf_counter() - start)
    return dict(
        solve_seconds=float(np.median(samples)), samples=samples, capture_warmup_seconds=warmup
    )


def eigsh(
    a,
    k=6,
    which="LM",
    ncv=None,
    tol=0,
    v0=None,
    maxiter=None,
    return_eigenvectors=True,
    sigma=None,
    **kwargs,
):
    import cupy as cp
    import pytest

    if kwargs:
        pytest.skip("Warpack has no CuPy API for: " + ", ".join(kwargs))
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected square matrix")
    n = a.shape[0]
    if a.dtype.kind not in "fc":
        raise TypeError("expected floating or complex matrix")
    if not 0 < k < n or which not in ("LM", "LA", "SA", "SM"):
        raise ValueError("invalid k or which")
    if np.dtype(a.dtype) not in (np.dtype("float64"), np.dtype("complex128")):
        pytest.skip("Warpack supports FP64/complex128 only")
    ah = host(a)
    if v0 is not None and np.iscomplexobj(ah) and (sigma is not None or which == "SM"):
        pytest.skip("Complex shift-invert has no initial_vector API")
    m = min(n, max(2 * k + 12, 32) if ncv is None else ncv)
    iterations = 300 if maxiter is None else maxiter
    if m <= k:
        pytest.skip("Warpack requires ncv > k; CuPy clamps this parameter")
    threshold = 1e-14 if tol == 0 else tol
    inverse = None
    evaluation = None
    auto_shift = which == "SM" and sigma is None
    if which == "SM" and sigma is not None:
        raise ValueError("SM cannot be combined with sigma")
    shifted = sigma is not None or auto_shift
    if shifted:
        if auto_shift:
            vals = ah.data if sp.issparse(ah) else ah
            scale = float(np.max(np.abs(vals))) if vals.size else 0.0
            sigma = -np.sqrt(np.finfo(np.float64).eps) * (scale if scale > 0 else 1.0)
        if np.iscomplexobj(ah) and which not in ("LM", "SM"):
            pytest.skip("complex shift-invert exposes nearest eigenvalues only")
        if np.iscomplexobj(ah):
            solver = ShiftInvertEigensolver(
                complex_matrix(ah), k, sigma, ncv=m, backend="cudss", tol=threshold
            )
            inverse = solver
        else:
            a0 = matrix(ah)
            inverse = CuDSSInverse(
                matrix(sp.csr_matrix(ah) - sigma * sp.eye(n)), 1, mtype="general", matching=True
            )
            solver = KrylovSchur(
                inverse,
                k,
                ncv=m,
                which="LM" if auto_shift else which,
                tol=threshold,
                initial_vector=None if v0 is None else wp.array(host(v0), dtype=wp.float64),
            )
            evaluation = EigenpairEvaluation(SparseOperator(a0), solver.eigenvectors, tol=threshold)
        iterations = 1 if m == n else min(iterations, 30)
    elif np.iscomplexobj(ah):
        start = None
        if v0 is not None:
            v0h = np.asarray(host(v0))
            if v0h.shape != (n,):
                raise ValueError("v0 must have shape (n,)")
            start = wp.array(np.column_stack((v0h.real, v0h.imag)), dtype=wp.vec2d)
        solver = HermitianEigensolver(
            complex_matrix(ah, True),
            k,
            which=which,
            tol=threshold,
            ncv=min(2 * n, 2 * m) if ncv is not None else None,
            initial_vector=start,
        )
    else:
        solver = KrylovSchur(
            SparseOperator(matrix(ah)),
            k,
            ncv=m,
            which=which,
            tol=threshold,
            initial_vector=None if v0 is None else wp.array(host(v0), dtype=wp.float64),
        )
    if not shifted and maxiter is None:
        # One cycle spans the entire space; repeated full-space iteration
        # cannot improve its roundoff floor. Assertions remain unchanged.
        if (np.iscomplexobj(ah) and solver.real.m == 2 * n) or (not np.iscomplexobj(ah) and m == n):
            iterations = 1
    try:
        options = {} if evaluation is None else {"finalizer": evaluation.run}
        execute(solver, iterations, **options)
        w = output(solver.eigenvalues if evaluation is None else evaluation.values).real
        v = output(solver.eigenvectors).T
        if shifted:
            order = (
                np.argsort(np.abs(w))[::-1]
                if auto_shift and not return_eigenvectors
                else np.argsort(w)
            )
            w, v = w[order], v[:, order]
        return (cp.asarray(w), cp.asarray(v)) if return_eigenvectors else cp.asarray(w)
    finally:
        # Destroy graph references before releasing backend storage.
        if hasattr(solver, "graph"):
            solver.graph = None
        if hasattr(solver, "core") and hasattr(solver.core, "graph"):
            solver.core.graph = None
        if inverse is not None:
            inverse.close()


def svds(a, k=6, ncv=None, tol=0, which="LM", maxiter=None, return_singular_vectors=True, **kwargs):
    import cupy as cp
    import pytest

    # rng=0 in upstream's ordinary fixtures only requests a reproducible start;
    # Warpack already uses a fixed seed. Other RNG semantics are tested separately.
    if kwargs.pop("rng", 0) != 0 or kwargs:
        pytest.skip("PartialSVD has no configurable rng/v0 API")
    if len(a.shape) != 2:
        raise ValueError("invalid shape")
    if a.dtype.kind not in "fc":
        raise TypeError("expected floating matrix")
    if not 0 < k < min(a.shape) or which != "LM":
        raise ValueError("invalid k or which")
    if np.dtype(a.dtype) != np.dtype("float64"):
        pytest.skip("PartialSVD supports real FP64 only")
    solver = PartialSVD(matrix(host(a)), k, ncv=ncv, tol=1e-14 if tol == 0 else tol)
    execute(solver, 300 if maxiter is None else maxiter)
    s = cp.asarray(output(solver.singular_values))
    if not return_singular_vectors:
        return s
    return cp.asarray(output(solver.u).T), s, cp.asarray(output(solver.v))
