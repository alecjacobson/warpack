"""Shared setup and independent acceptance checks for application comparisons."""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.upstream.adapters import complex_matrix, matrix, output
from warpack import (
    BucklingEigensolver,
    CGInverse,
    CuDSSInverse,
    GeneralizedEigensolver,
    GeneralizedKrylovSchur,
    PartialSVD,
    ShiftInvertEigensolver,
    SparseOperator,
)


def reference_values(p):
    if p.expected is not None:
        return np.asarray(p.expected)
    if p.n <= 512:
        vals = la.eigvalsh(p.a.toarray(), p.b.toarray())
        return vals[np.argsort(-abs(vals))[: p.k]]
    return sla.eigsh(p.a, k=p.k, M=p.b, which="LM", tol=1e-11, return_eigenvectors=False)


def accuracy(p, values, vectors, right=None, reference=None):
    """Compare physical equations and requested eigenvalues, outside graph capture."""
    ref = reference_values(p) if reference is None else reference
    cost = abs(values[:, None] - ref[None, :]) / np.maximum(1, abs(ref))[None, :]
    ii, jj = opt.linear_sum_assignment(cost)
    value_error = float(cost[ii, jj].max())
    if p.mode == "svd":
        u, v = vectors, right
        av, atu = p.a @ v, p.a.T @ u
        r1, r2 = av - u * values, atu - v * values
        residual = max(np.linalg.norm(r1, axis=0).max(), np.linalg.norm(r2, axis=0).max()) / max(
            1, float(max(values))
        )
        orth = max(abs(u.T @ u - np.eye(p.k)).max(), abs(v.T @ v - np.eye(p.k)).max())
        return dict(
            accepted=bool(residual < 1e-8 and orth < 1e-8 and value_error < 1e-8),
            residual=float(residual),
            orthogonality=float(orth),
            spectrum_error=value_error,
            captured_energy=float(sum(values**2) / sum(p.a.data**2)),
        )
    av = p.a @ vectors
    bv = vectors if p.b is None else p.b @ vectors
    r = av - bv * values
    residual = np.linalg.norm(r, axis=0) / np.maximum(
        1, np.linalg.norm(av, axis=0) + abs(values) * np.linalg.norm(bv, axis=0)
    )
    if p.api_class == "H":
        orth = None  # Nonnormal eigenvectors need not be orthogonal.
        norms = np.linalg.norm(vectors, axis=0)
        normalized = bool(np.max(abs(norms - 1)) < 1e-8)
    else:
        metric_v = p.a @ vectors if p.mode == "buckling" else bv
        orth = float(abs(vectors.conj().T @ metric_v - np.eye(p.k)).max())
        normalized = orth < 1e-8
    result = dict(
        accepted=bool(residual.max() < 1e-8 and value_error < 1e-8 and normalized),
        residual=float(residual.max()),
        max_absolute_residual=float(abs(r).max()),
        orthogonality=orth,
        spectrum_error=value_error,
    )
    if p.nullity or p.api_class == "D" or p.null_basis is not None:
        result["recovered_nullity"] = int(np.count_nonzero(abs(values) < 1e-8))
        result["accepted"] &= result["recovered_nullity"] == p.nullity
    if p.api_class == "D":
        positive = values[values > 1e-8]
        result["positive_guard"] = float(positive.min()) if len(positive) else None
        if p.require_positive_guard:
            result["accepted"] &= result["positive_guard"] is not None
    if p.null_basis is not None and result["recovered_nullity"] == p.nullity:
        zeros = vectors[:, abs(values) < 1e-8]
        overlap = p.null_basis.T @ (p.b @ zeros)
        remainder = zeros - p.null_basis @ overlap
        result["nullspace_subspace_error"] = float(
            np.sqrt(max(0.0, np.sum(remainder * (p.b @ remainder))))
        )
        result["accepted"] &= result["nullspace_subspace_error"] < 1e-8
    if p.requested_basis is not None:
        basis = p.requested_basis
        remainder = vectors - basis @ (basis.conj().T @ bv)
        metric_remainder = remainder if p.b is None else p.b @ remainder
        result["requested_subspace_error"] = float(
            np.sqrt(max(0.0, np.sum(remainder.conj() * metric_remainder).real))
        )
        result["accepted"] &= result["requested_subspace_error"] < 1e-8
    return result


def warp_setup(p, method="krylov", ncv=32, tol=1e-11, inverse_transform=None):
    ncv = min(ncv, p.n)
    if p.api_class in "EH":
        solver = ShiftInvertEigensolver(complex_matrix(p.a), p.k, p.target, ncv=ncv, tol=tol)
        return solver, solver
    if p.mode == "svd":
        return PartialSVD(matrix(p.a), p.k, ncv=ncv, tol=tol), None
    a, b = matrix(p.a), matrix(p.b)
    shifted = p.b if p.mode == "regular" else p.a - p.target * p.b
    block = method == "block" or p.api_class in "DG"
    width = ncv if block else 1
    if method.startswith("cg"):
        if p.api_class not in "ACD" or p.mode != "shift-invert":
            raise ValueError("CG benchmark only supplies known SPD shifted systems")
        if method == "cg":
            shifted_device = matrix(shifted)
            preconditioner = "diag"
        else:
            blocks = shifted.tobsr(blocksize=(p.block_size, p.block_size))
            dtype = wp.types.matrix(shape=(p.block_size, p.block_size), dtype=wp.float64)
            shifted_device = ws.bsr_from_triplets(
                p.n // p.block_size,
                p.n // p.block_size,
                wp.array(
                    np.repeat(np.arange(p.n // p.block_size), np.diff(blocks.indptr)),
                    dtype=wp.int32,
                ),
                wp.array(blocks.indices, dtype=wp.int32),
                wp.array(blocks.data, dtype=dtype),
            )
            if method == "cg-bj":
                preconditioner = "block_jacobi_sequential"
            elif method == "cg-fsai":
                preconditioner = linear.FSAI(
                    shifted_device,
                    max_row_size=32,
                    max_step_size=3,
                    apply_lanes=8,
                    factor_dtype=wp.float32,
                )
            else:
                raise ValueError(method)
        inv = CGInverse(shifted_device, tol=1e-13, maxiter=10000, preconditioner=preconditioner)
    else:
        inv = CuDSSInverse(
            matrix(shifted),
            width,
            mtype="spd" if p.api_class in "ACDF" else "general",
            matching=p.api_class not in "ACDF",
        )
    transformed = inv
    if inverse_transform is not None:
        if block or p.mode != "shift-invert":
            raise ValueError("Inverse transformation requires one-RHS generalized shift-invert")
        transformed = inverse_transform(inv, SparseOperator(b))
    if p.mode == "buckling":
        solver = BucklingEigensolver(
            SparseOperator(a), SparseOperator(b), inv, p.k, p.target, ncv=ncv, tol=tol
        )
    elif block:
        if p.mode == "regular":
            inv.close()
            raise ValueError("old block API has no regular mass-inverse mode")
        solver = GeneralizedEigensolver(
            SparseOperator(a), SparseOperator(b), inv, p.k, ncv=ncv, target=p.target, tol=tol
        )
    else:
        solver = GeneralizedKrylovSchur(
            SparseOperator(a),
            SparseOperator(b),
            transformed,
            p.k,
            ncv=ncv,
            target=p.target,
            mode=p.mode,
            tol=tol,
        )
    return solver, inv


def warp_output(p, solver):
    if p.mode == "svd":
        return output(solver.singular_values), output(solver.u).T, output(solver.v).T
    return output(solver.eigenvalues), output(solver.eigenvectors).T, None


def scipy_setup(p, ncv=None):
    """Factorization outside solve timing; every solve restores the same start."""
    rng = np.random.default_rng(2026)
    initial = rng.normal(size=p.n)
    if p.mode == "svd":

        def solve():
            u, w, vt = sla.svds(p.a, k=p.k, which="LM", tol=1e-11, random_state=2026, ncv=ncv)
            return w, u, vt.T

        return solve
    if p.mode == "regular":
        lu = sla.splu(p.b.tocsc())
        mi = sla.LinearOperator(p.b.shape, matvec=lu.solve)

        def solve():
            w, v = sla.eigsh(
                p.a, k=p.k, M=p.b, Minv=mi, which="LM", tol=1e-11, v0=initial.copy(), ncv=ncv
            )
            return w, v, None

        return solve
    shifted = p.a - p.target * (sp.eye(p.n) if p.b is None else p.b)
    lu = sla.splu(shifted.tocsc())
    op = sla.LinearOperator(shifted.shape, matvec=lu.solve, dtype=shifted.dtype)

    def solve():
        kwargs = dict(
            k=p.k, sigma=p.target, OPinv=op, which="LM", tol=1e-11, v0=initial.copy(), ncv=ncv
        )
        if p.api_class == "H":
            w, v = sla.eigs(p.a, **kwargs)
        else:
            if p.b is not None:
                kwargs["M"] = p.b
            if p.mode in ("buckling", "cayley"):
                kwargs["mode"] = p.mode
            w, v = sla.eigsh(p.a, **kwargs)
        return w, v, None

    return solve
