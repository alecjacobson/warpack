"""Real or complex shift-invert for general complex CSR, on device."""

import warp as wp
import warp.sparse as ws
from warp.optim import linear

from .arnoldi import (
    GeneralEigensolver,
    complex_outputs,
    complex_residual,
)
from .kernels import identity_order
from .schur import cdiv
from .solver import CuDSSInverse


@wp.kernel
def realify_triplets(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    a: wp.array[wp.vec2d],
    n: int,
    nnz: int,
    sigma: wp.vec2d,
    rows_out: wp.array[wp.int32],
    cols_out: wp.array[wp.int32],
    values: wp.array[wp.float64],
):
    i = wp.tid()
    for e in range(offsets[i], offsets[i + 1]):
        j = cols[e]
        v = a[e]
        rows_out[4 * e] = i
        cols_out[4 * e] = j
        values[4 * e] = v[0]
        rows_out[4 * e + 1] = i
        cols_out[4 * e + 1] = j + n
        values[4 * e + 1] = -v[1]
        rows_out[4 * e + 2] = i + n
        cols_out[4 * e + 2] = j
        values[4 * e + 2] = v[1]
        rows_out[4 * e + 3] = i + n
        cols_out[4 * e + 3] = j + n
        values[4 * e + 3] = v[0]
    e = 4 * nnz + 4 * i
    rows_out[e] = i
    cols_out[e] = i
    values[e] = -sigma[0]
    rows_out[e + 1] = i
    cols_out[e + 1] = i + n
    values[e + 1] = sigma[1]
    rows_out[e + 2] = i + n
    cols_out[e + 2] = i
    values[e + 2] = -sigma[1]
    rows_out[e + 3] = i + n
    cols_out[e + 3] = i + n
    values[e + 3] = -sigma[0]


@wp.kernel
def unpack_complex(x: wp.array2d[wp.vec2d], out: wp.array2d[wp.float64], j: int, n: int):
    i = wp.tid()
    out[0, i] = x[j, i][0]
    out[0, i + n] = x[j, i][1]


@wp.kernel
def pack_complex(x: wp.array2d[wp.float64], out: wp.array2d[wp.vec2d], j: int, n: int):
    i = wp.tid()
    out[j, i] = wp.vec2d(x[0, i], x[0, i + n])


@wp.kernel
def unshift(values: wp.array[wp.vec2d], out: wp.array[wp.vec2d], sigma: wp.vec2d):
    i = wp.tid()
    out[i] = sigma + cdiv(wp.vec2d(1.0, 0.0), values[i])


class ComplexShiftInverse:
    """Apply (A-sigma I)^-1 using cuDSS or pure-Warp GMRES.

    The complex matrix is represented as a real 2n system. Initial sparse
    topology/factor setup stays outside capture. Applies support multiple RHS.
    """

    requires_static_outer_capture = True

    def __init__(self, operator, sigma=0j, backend="cudss", inner_tol=1.0e-12, inner_maxiter=1000):
        if backend not in ("cudss", "gmres"):
            raise ValueError("backend must be cudss or gmres")
        self.original = operator
        self.complex_n = operator.complex_n
        self.device = operator.device
        n = self.complex_n
        nnz = len(operator.values)
        size = 4 * (nnz + n)
        rows = wp.empty(size, dtype=wp.int32, device=self.device)
        cols = wp.empty_like(rows)
        vals = wp.empty(size, dtype=wp.float64, device=self.device)
        self.sigma = wp.vec2d(float(complex(sigma).real), float(complex(sigma).imag))
        wp.launch(
            realify_triplets,
            n,
            [
                operator.offsets,
                operator.columns,
                operator.values,
                n,
                nnz,
                self.sigma,
                rows,
                cols,
                vals,
            ],
            device=self.device,
        )
        self.matrix = ws.bsr_from_triplets(2 * n, 2 * n, rows, cols, vals)
        self.rhs = wp.zeros((1, 2 * n), dtype=wp.float64, device=self.device)
        self.solution = wp.empty_like(self.rhs)
        self.backend = backend
        if backend == "cudss":
            self.inverse = CuDSSInverse(self.matrix, 1, mtype="general")
        else:
            self.state = linear.gmres(
                self.matrix,
                self.rhs[0],
                self.solution[0],
                tol=inner_tol,
                maxiter=inner_maxiter,
                restart=min(2 * n, 64),
                check_every=0,
                run=False,
            )

    def apply_complex(self, x, y):
        for j in range(x.shape[0]):
            wp.launch(
                unpack_complex,
                self.complex_n,
                [x, self.rhs, j, self.complex_n],
                device=self.device,
            )
            if self.backend == "cudss":
                self.inverse.apply(self.rhs, self.solution)
            else:
                self.solution.zero_()
                self.state(b=self.rhs[0], x=self.solution[0])
            wp.launch(
                pack_complex,
                self.complex_n,
                [self.solution, y, j, self.complex_n],
                device=self.device,
            )

    def close(self):
        if self.backend == "cudss":
            self.inverse.close()


class ShiftInvertEigensolver:
    """Eigenpairs nearest a real or complex sigma; original-problem residuals."""

    def __init__(
        self,
        operator,
        k,
        sigma=0j,
        ncv=None,
        backend="cudss",
        tol=1.0e-9,
        inner_tol=1.0e-12,
        inner_maxiter=1000,
    ):
        self.inverse = ComplexShiftInverse(operator, sigma, backend, inner_tol, inner_maxiter)
        self.core = GeneralEigensolver(self.inverse, k, ncv=ncv, which="LM", tol=tol * 0.01)
        self.op = operator
        self.device = operator.device
        self.k = k
        self.tol = tol
        self.eigenvectors = self.core.eigenvectors
        self.eigenvalues = wp.empty_like(self.core.eigenvalues)
        self.residuals = wp.empty(k, dtype=wp.float64, device=self.device)
        self.norms = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.ax = wp.empty_like(self.eigenvectors)
        self.order = wp.empty(k, dtype=wp.int32, device=self.device)
        self.converged = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.value_buffer = wp.empty_like(self.eigenvalues)

    def finalize(self):
        wp.launch(
            unshift,
            self.k,
            [self.core.eigenvalues, self.value_buffer, self.inverse.sigma],
            device=self.device,
        )
        wp.launch(identity_order, self.k, [self.order], device=self.device)
        self.op.apply_complex(self.eigenvectors, self.ax)
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (self.op.complex_n + 255) // 256),
            inputs=[
                self.eigenvectors,
                self.ax,
                self.value_buffer,
                self.order,
                self.norms,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            complex_outputs,
            self.k,
            [
                self.value_buffer,
                self.order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )

    def solve(self, iterations=10):
        self.core.solve(iterations)
        self.finalize()
        return self

    def capture(self, iterations=10):
        self.graph = self.core.capture(iterations, adaptive=False, finalizer=self.finalize)
        return self.graph

    def close(self):
        self.inverse.close()
