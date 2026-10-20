"""Reference transforms are explicit; no singular metric enters LOBPCG."""

import numpy as np
import scipy.sparse.linalg as sla
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse

from .problems import Problem
from .references import spectra_setup


def transformed(case):
    k, m = case.reduced
    return Problem("conformal_reciprocal", "F", m, k - case.shift * m, case.k, 0.0, mode="regular")


def scipy_setup(case, ncv, original=False):
    k, m = case.reduced
    a = k - case.shift * m
    factor = sla.splu(a.tocsc())
    inverse = sla.LinearOperator(a.shape, matvec=factor.solve)
    initial = np.random.default_rng(2026).normal(size=case.n)

    def solve():
        if original:
            values, vectors = sla.eigsh(
                k,
                M=m,
                sigma=case.shift,
                OPinv=inverse,
                k=case.k,
                ncv=ncv,
                which="LM",
                tol=1e-11,
                v0=initial.copy(),
            )
        else:
            mu, vectors = sla.eigsh(
                m, M=a, Minv=inverse, k=case.k, ncv=ncv, which="LA", tol=1e-11, v0=initial.copy()
            )
            values = case.shift + 1 / mu
            vectors /= np.sqrt(mu)
        return values, case.lift(vectors), None

    return solve, lambda: None


def spectra_reciprocal(case, ncv):
    solve_native, close = spectra_setup(transformed(case), ncv)

    def solve():
        mu, vectors, _ = solve_native()
        return case.shift + 1 / mu, case.lift(vectors / np.sqrt(mu)), None

    return solve, close


def cupy_reciprocal(case, tolerance=1e-9, maxiter=300, scaled=False):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    p = transformed(case)
    scale = -case.shift if scaled else 1.0
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        mass, metric = cs.csr_matrix(scale * p.a), cs.csr_matrix(p.b)
        inverse = CuDSSInverse(matrix(p.b), 1, refinement_steps=1)
        output = cp.asarray(inverse.x)
        free, boundary = cp.asarray(case.free), cp.asarray(case.boundary)
        initial = cp.asarray(np.random.default_rng(2026).normal(size=(case.n, case.k)))

        def apply(x):
            with stream:
                xx = x[:, None] if x.ndim == 1 else x
                out = cp.empty_like(xx)
                for j in range(xx.shape[1]):
                    rhs = cp.ascontiguousarray(xx[:, j]).reshape(1, case.n)
                    inverse.apply(wp.from_dlpack(rhs), inverse.x)
                    out[:, j] = output[0]
                return out[:, 0] if x.ndim == 1 else out

        pre = cl.LinearOperator((case.n, case.n), matvec=apply, matmat=apply, dtype=cp.float64)

    def solve():
        with stream:
            cp.random.seed(2026)
            mu, vectors, history = cl.lobpcg(
                mass,
                initial.copy(),
                B=metric,
                M=pre,
                largest=True,
                tol=tolerance,
                maxiter=maxiter,
                retResidualNormsHistory=True,
            )
            fresh = cp.linalg.norm(mass @ vectors - (metric @ vectors) * mu, axis=0)
            out = cp.zeros((2 * len(case.xyz), case.k))
            out[free] = vectors / cp.sqrt(mu / scale)
            n = len(case.xyz)
            for offset in (0, n):
                out[offset : offset + n] -= out[boundary + offset].mean(axis=0)
            return case.shift + scale / mu, out, (history, fresh)

    return solve, inverse.close
