"""Largest singular triplets via a symmetric augmented operator, without A^T A."""

import warp as wp
import warp.sparse as ws

from . import kernels as K
from .solver import KrylovSchur


@wp.kernel
def augmented_mv(
    ar: wp.array[wp.int32],
    ac: wp.array[wp.int32],
    av: wp.array[wp.float64],
    br: wp.array[wp.int32],
    bc: wp.array[wp.int32],
    bv: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
    nrow: int,
):
    j, i = wp.tid()
    val = wp.float64(0.0)
    if i < nrow:
        for e in range(ar[i], ar[i + 1]):
            val += av[e] * x[j, nrow + ac[e]]
    else:
        r = i - nrow
        for e in range(br[r], br[r + 1]):
            val += bv[e] * x[j, bc[e]]
    y[j, i] = val


@wp.kernel
def split_vectors(
    q: wp.array2d[wp.float64],
    u: wp.array2d[wp.float64],
    v: wp.array2d[wp.float64],
    vals: wp.array[wp.float64],
    out: wp.array[wp.float64],
    nrow: int,
):
    j, i = wp.tid()
    if i < nrow:
        u[j, i] = q[j, i]
    else:
        v[j, i - nrow] = q[j, i]
    if i == 0:
        out[j] = wp.max(wp.float64(0.0), vals[j])


@wp.kernel
def rect_mv(
    offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    j, i = wp.tid()
    val = wp.float64(0.0)
    for e in range(offsets[i], offsets[i + 1]):
        val += values[e] * x[j, cols[e]]
    y[j, i] = val


class _Augmented:
    def __init__(self, a):
        if a.scalar_type != wp.float64 or a.block_shape != (1, 1) or a.row_counts is not None:
            raise ValueError("SVD needs compact scalar FP64 CSR")
        self.a = a
        self.at = ws.bsr_transposed(a)
        self.n = sum(a.shape)
        self.device = a.device

    def apply(self, x, y):
        a, b = self.a, self.at
        wp.launch(
            augmented_mv,
            x.shape,
            [a.offsets, a.columns, a.values, b.offsets, b.columns, b.values, x, y, a.shape[0]],
            device=self.device,
        )


class PartialSVD:
    """Largest k singular triplets of rectangular real FP64 CSR.

    Left/right vectors use row-major (k,nrow)/(k,ncol) storage. An augmented
    symmetric operator avoids squaring the condition number. Always inspect
    both left and right residuals; zero singular subspaces are not unique.
    """

    def __init__(self, matrix, k, ncv=None, tol=1.0e-10):
        if not 1 <= k <= min(matrix.shape):
            raise ValueError("require 1 <= k <= min(matrix.shape)")
        self.op = _Augmented(matrix)
        self.device = matrix.device
        self.k = k
        self.core = KrylovSchur(
            self.op, k, ncv=ncv or min(self.op.n, max(4 * k, 48)), which="LA", tol=tol
        )
        self.u = wp.empty((k, matrix.shape[0]), dtype=wp.float64, device=self.device)
        self.v = wp.empty((k, matrix.shape[1]), dtype=wp.float64, device=self.device)
        self.singular_values = wp.empty(k, dtype=wp.float64, device=self.device)
        self.dots = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.norm = wp.zeros((1, 1), dtype=wp.float64, device=self.device)
        self.av = wp.empty_like(self.u)
        self.atu = wp.empty_like(self.v)
        self.left_norms = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.right_norms = wp.zeros_like(self.left_norms)
        self.left_residuals = wp.empty(k, dtype=wp.float64, device=self.device)
        self.right_residuals = wp.empty_like(self.left_residuals)
        self.order = wp.empty(k, dtype=wp.int32, device=self.device)
        self.count_left = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.count_right = wp.zeros_like(self.count_left)
        self.value_scratch = wp.empty_like(self.singular_values)
        self.tol = tol

    def finalize(self):
        wp.launch(
            split_vectors,
            self.core.eigenvectors.shape,
            [
                self.core.eigenvectors,
                self.u,
                self.v,
                self.core.eigenvalues,
                self.singular_values,
                self.u.shape[1],
            ],
            device=self.device,
        )
        for vectors in (self.u, self.v):
            n = vectors.shape[1]
            for j in range(self.k):
                if j:
                    for _ in range(2):
                        self.dots.zero_()
                        wp.launch_tiled(
                            K.basis_dots,
                            dim=(j, (n + 255) // 256),
                            inputs=[vectors, self.dots, j],
                            block_dim=128,
                            device=self.device,
                        )
                        wp.launch(K.basis_subtract, n, [vectors, self.dots, j], device=self.device)
                self.norm.zero_()
                wp.launch_tiled(
                    K.basis_norm,
                    dim=(n + 255) // 256,
                    inputs=[vectors, self.norm, j],
                    block_dim=128,
                    device=self.device,
                )
                wp.launch(K.basis_scale, n, [vectors, self.norm, j], device=self.device)
        a, b = self.op.a, self.op.at
        wp.launch(
            rect_mv,
            self.av.shape,
            [a.offsets, a.columns, a.values, self.v, self.av],
            device=self.device,
        )
        wp.launch(
            rect_mv,
            self.atu.shape,
            [b.offsets, b.columns, b.values, self.u, self.atu],
            device=self.device,
        )
        wp.launch(K.identity_order, self.k, [self.order], device=self.device)
        for vec, prod, norms, errors, count in (
            (self.u, self.av, self.left_norms, self.left_residuals, self.count_left),
            (self.v, self.atu, self.right_norms, self.right_residuals, self.count_right),
        ):
            norms.zero_()
            count.zero_()
            wp.launch_tiled(
                K.residual,
                dim=(self.k, (vec.shape[1] + 255) // 256),
                inputs=[vec, prod, self.singular_values, self.order, norms],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                K.output_values,
                self.k,
                [
                    self.singular_values,
                    self.order,
                    self.value_scratch,
                    norms,
                    errors,
                    self.tol,
                    count,
                ],
                device=self.device,
            )

    def capture(self, max_iterations=100):
        self.graph = self.core.capture(max_iterations, finalizer=self.finalize)
        return self.graph

    def solve(self, iterations=30):
        self.core.solve(iterations)
        self.finalize()
        return self
