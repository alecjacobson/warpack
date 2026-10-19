"""Experimental largest nonzero SVD triplets using matrix-free normal products.

This deliberately squares conditioning; the augmented production method remains
unchanged. Both original SVD equations must pass, and unresolved/zero requested
singular values are rejected rather than producing a false full count.
"""

import math

import warp as wp

from warpack import kernels as K
from warpack.svd import _LongRowProduct, _SVDKrylovSchur, _Transpose, rect_mv, rect_mv_cooperative


class Product:
    def __init__(self, matrix, width):
        self.matrix = matrix
        self.long = _LongRowProduct(matrix, width) if matrix.nnz / matrix.nrow > 4096 else None

    def apply(self, x, y):
        a = self.matrix
        if self.long is not None:
            self.long.apply(x, y)
        elif a.nnz / a.nrow > 64:
            wp.launch_tiled(
                rect_mv_cooperative,
                dim=y.shape,
                inputs=[a.offsets, a.columns, a.values, x, y],
                block_dim=32,
                device=a.device,
            )
        else:
            wp.launch(rect_mv, y.shape, [a.offsets, a.columns, a.values, x, y], device=a.device)


class NormalOperator:
    def __init__(self, a, width):
        if (
            a.scalar_type != wp.float64
            or a.block_shape != (1, 1)
            or a.row_counts is not None
            or min(a.shape) < 1
        ):
            raise ValueError("require nonempty compact scalar FP64 CSR")
        self.a = a
        self.transpose = _Transpose(a)
        self.at = self.transpose.matrix
        self.n, self.device = min(a.shape), a.device
        self.tall = a.shape[0] >= a.shape[1]
        self.forward, self.backward = Product(a, width), Product(self.at, width)
        self.tmp = wp.empty((width, max(a.shape)), dtype=wp.float64, device=a.device)

    def refresh(self):
        self.transpose.refresh()

    def apply(self, x, y):
        if x.shape[0] > self.tmp.shape[0]:
            raise ValueError("RHS count exceeds preallocated width")
        tmp = self.tmp[: x.shape[0]]
        first, second = (
            (self.forward, self.backward) if self.tall else (self.backward, self.forward)
        )
        first.apply(x, tmp)
        second.apply(tmp, y)


@wp.kernel
def normalize_lift(
    vectors: wp.array2d[wp.float64],
    norms: wp.array2d[wp.float64],
    singular_values: wp.array[wp.float64],
):
    j, i = wp.tid()
    sigma = wp.sqrt(wp.max(wp.float64(0.0), norms[j, 0]))
    if sigma > wp.float64(0.0):
        vectors[j, i] /= sigma
    if i == 0:
        singular_values[j] = sigma


@wp.kernel
def reject_unresolved(
    eigenvalues: wp.array[wp.float64],
    norms: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    left_count: wp.array[wp.int32],
    right_count: wp.array[wp.int32],
):
    # Normal equations cannot reliably resolve eigenvalues at the roundoff floor
    # of the largest one. Do not invent orthonormal zero singular vectors.
    floor = wp.max(wp.abs(eigenvalues[0]), wp.float64(1.0e-300)) * wp.float64(1.0e-14)
    for j in range(eigenvalues.shape[0]):
        if (
            not wp.isfinite(eigenvalues[j])
            or eigenvalues[j] <= floor
            or not wp.isfinite(norms[j, 0])
            or norms[j, 0] <= floor
        ):
            status[0] = 2
            left_count[0] = 0
            right_count[0] = 0


@wp.kernel
def check_orthogonality(
    gram: wp.array2d[wp.float64],
    errors: wp.array[wp.float64],
    side: int,
    tolerance: wp.float64,
    status: wp.array[wp.int32],
    left_count: wp.array[wp.int32],
    right_count: wp.array[wp.int32],
):
    error = wp.float64(0.0)
    for i in range(gram.shape[0]):
        for j in range(gram.shape[1]):
            expected = wp.float64(0.0)
            if i == j:
                expected = wp.float64(1.0)
            value = wp.abs(gram[i, j] - expected)
            if not wp.isfinite(value):
                value = wp.float64(wp.inf)
            error = wp.max(error, value)
    errors[side] = error
    if error > tolerance:
        # Preserve earlier rank/conditioning failures while marking otherwise
        # residual-passing triplets unsuitable as an orthonormal POD basis.
        if status[0] == 0:
            status[0] = 3
        left_count[0] = 0
        right_count[0] = 0


class NormalSVD:
    """Prototype for well-resolved largest nonzero triplets; fixed sparse topology.

    ncv can equal min(matrix.shape), giving a complete normal-operator basis.
    The existing independent-chain guard is unchanged for partial bases.
    status=2 rejects requested zero/roundoff-scale values; it is not a rank
    estimate. status=3 rejects triplets whose left/right Gram matrices differ
    entrywise from identity by more than orthogonality_tol (default 1e-8).
    Both rejection statuses clear accepted-triplet counts; equation residuals
    remain available. Matrix values can change in place before captured replay.
    """

    def __init__(self, matrix, k, ncv=None, tol=1e-11, orthogonality_tol=1e-8):
        if not math.isfinite(orthogonality_tol) or orthogonality_tol <= 0:
            raise ValueError("orthogonality_tol must be finite and positive")
        if not 1 <= k < min(matrix.shape):
            raise ValueError("require 1 <= k < min(matrix.shape)")
        ncv = min(min(matrix.shape), ncv or max(4 * k, 48))
        self.op = NormalOperator(matrix, max(k, ncv))
        self.core = _SVDKrylovSchur(self.op, k, ncv=ncv, which="LA", tol=tol)
        self.k, self.device, self.tol = k, matrix.device, tol
        self.u = wp.empty((k, matrix.shape[0]), dtype=wp.float64, device=self.device)
        self.v = wp.empty((k, matrix.shape[1]), dtype=wp.float64, device=self.device)
        self.av, self.atu = wp.empty_like(self.u), wp.empty_like(self.v)
        self.singular_values = wp.empty(k, dtype=wp.float64, device=self.device)
        self.lift_norms = wp.zeros((k, 1), dtype=wp.float64, device=self.device)
        self.left_norms, self.right_norms = (
            wp.zeros_like(self.lift_norms),
            wp.zeros_like(self.lift_norms),
        )
        self.left_residuals = wp.empty_like(self.singular_values)
        self.right_residuals = wp.empty_like(self.singular_values)
        self.count_left = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.count_right = wp.zeros_like(self.count_left)
        self.order = wp.empty(k, dtype=wp.int32, device=self.device)
        self.scratch_values = wp.empty_like(self.singular_values)
        self.orthogonality_tol = orthogonality_tol
        self.orthogonality_errors = wp.empty(2, dtype=wp.float64, device=self.device)
        self.gram = wp.empty((k, k), dtype=wp.float64, device=self.device)

    def finalize(self):
        small, large = (self.v, self.u) if self.op.tall else (self.u, self.v)
        wp.copy(small, self.core.eigenvectors)
        product = self.op.forward if self.op.tall else self.op.backward
        product.apply(small, large)
        self.lift_norms.zero_()
        # Recompute singular values from the original product, avoiding sqrt of
        # a slightly inaccurate normal-equation Ritz value.
        for j in range(self.k):
            wp.launch_tiled(
                K.basis_norm,
                dim=(large.shape[1] + 255) // 256,
                inputs=[large, self.lift_norms[j : j + 1], j],
                block_dim=128,
                device=self.device,
            )
        wp.launch(
            normalize_lift,
            large.shape,
            [large, self.lift_norms, self.singular_values],
            device=self.device,
        )
        self.op.forward.apply(self.v, self.av)
        self.op.backward.apply(self.u, self.atu)
        wp.launch(K.identity_order, self.k, [self.order], device=self.device)
        for vectors, products, norms, residuals, count in (
            (self.u, self.av, self.left_norms, self.left_residuals, self.count_left),
            (self.v, self.atu, self.right_norms, self.right_residuals, self.count_right),
        ):
            norms.zero_()
            count.zero_()
            wp.launch_tiled(
                K.residual,
                dim=(self.k, (vectors.shape[1] + 255) // 256),
                inputs=[vectors, products, self.singular_values, self.order, norms],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                K.output_values,
                self.k,
                [
                    self.singular_values,
                    self.order,
                    self.scratch_values,
                    norms,
                    residuals,
                    self.tol,
                    count,
                ],
                device=self.device,
            )
        wp.launch(
            reject_unresolved,
            1,
            [
                self.core.eigenvalues,
                self.lift_norms,
                self.core.status,
                self.count_left,
                self.count_right,
            ],
            device=self.device,
        )

        for side, vectors in enumerate((self.u, self.v)):
            self.gram.zero_()
            wp.launch_tiled(
                K.gram_partial,
                dim=(self.k, self.k, (vectors.shape[1] + 255) // 256),
                inputs=[vectors, vectors, self.gram],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                check_orthogonality,
                1,
                [
                    self.gram,
                    self.orthogonality_errors,
                    side,
                    self.orthogonality_tol,
                    self.core.status,
                    self.count_left,
                    self.count_right,
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
