"""Pure-Warp two-level preconditioner experiment for #134; not a public API.

For SPD A, full-row-rank Z and SPD fixed-linear smoother S, Q = Z.T (Z A Z.T)^-1 Z.
Additive: B = S + Q. Balanced: B = Q + (I-Q A) S (I-A Q).
The balanced form is symmetric positive definite; neither form changes the
outer eigenproblem. No reference eigenvectors, CPU solves or cuDSS are used.
"""

import warp as wp
from warp.optim import linear

from warpack import kernels as K
from warpack._coalesced_fsai import CoalescedFactorApplication
from warpack.batched_cg import RepeatedOperator, accumulate_statistics, axpby
from warpack.block import gram
from warpack.block_products import subtract_projection
from warpack.generalized import _cg_preallocates_loop_control


@wp.kernel
def _check_pivots(
    g: wp.array2d[wp.float64], factor: wp.array2d[wp.float64], status: wp.array[wp.int32]
):
    i = wp.tid()
    if factor[i, i] * factor[i, i] <= wp.float64(1e-12) * wp.abs(g[i, i]):
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _subtract_coefficients(
    first: wp.array2d[wp.float64], second: wp.array2d[wp.float64], balanced: bool
):
    i, j = wp.tid()
    value = -first[i, j]
    if balanced:
        value += second[i, j]
    first[i, j] = value


@wp.kernel
def _reject_invalid(y: wp.array2d[wp.float64], status: wp.array[wp.int32]):
    i, j = wp.tid()
    if status[0] != 0:
        y[i, j] = wp.float64(wp.nan)


class FixedFSAI:
    """Use upstream FSAI factors as one fixed-linear FP64 block application."""

    def __init__(self, factor, width, dtype=wp.float64):
        self.n, self.device = factor.shape[0], factor.device
        self.factor = factor
        self.application = CoalescedFactorApplication(factor, width, self.n, self.device, dtype)

    def apply(self, x, y):
        self.application.matvec(x.flatten(), y.flatten(), y.flatten(), 1.0, 0.0)


class CoarseCorrection:
    """Caller-managed fixed-address basis/matrix refresh, capturable update and apply.

    ``status`` must be zero after each update/replay. A singular or numerically
    rank-deficient coarse space is rejected, not regularized. Fixed-linear SPD
    smoothers are required for the stated symmetry/positivity contract.
    """

    def __init__(self, operator, basis, smoother, width, *, balanced=True):
        self.n, self.device = operator.n, operator.device
        if (
            basis.dtype != wp.float64
            or basis.ndim != 2
            or basis.shape[1] != self.n
            or basis.device != self.device
        ):
            raise ValueError("coarse basis must be FP64 rows on the operator device")
        if not 0 < basis.shape[0] <= self.n or width < 1:
            raise ValueError("nonempty coarse basis and positive RHS width required")
        if smoother.n != self.n or smoother.device != self.device:
            raise ValueError("smoother must match operator shape and device")
        self.operator, self.basis, self.smoother = operator, basis, smoother
        self.width, self.balanced = width, bool(balanced)
        q = basis.shape[0]
        self.c = wp.empty_like(basis)
        self.d = wp.empty_like(basis)
        self.g = wp.zeros((q, q), dtype=wp.float64, device=self.device)
        self.l = wp.zeros_like(self.g)
        self.status = wp.ones(1, dtype=wp.int32, device=self.device)
        self.coeff = wp.zeros((width, q), dtype=wp.float64, device=self.device)
        self.second = wp.zeros_like(self.coeff)
        self.work = wp.empty((width, self.n), dtype=wp.float64, device=self.device)

    def update(self):
        self.status.zero_()
        self.operator.apply(self.basis, self.d)
        gram(self.basis, self.d, self.g, chunk=4096, symmetric=True)
        wp.launch(K.cholesky, 1, [self.g, self.l, self.status], device=self.device)
        wp.launch(
            _check_pivots, self.basis.shape[0], [self.g, self.l, self.status], device=self.device
        )
        wp.copy(self.c, self.basis)
        wp.launch(K.triangular, self.n, [self.c, self.l], device=self.device)
        # D = C A; same left solve as C = L^-1 Z, without a second sparse product.
        wp.launch(K.triangular, self.n, [self.d, self.l], device=self.device)

    def _subtract(self, basis, coefficients, target):
        wp.launch_tiled(
            subtract_projection,
            dim=((self.width + 3) // 4, (self.n + 31) // 32),
            inputs=[basis, coefficients, target],
            block_dim=128,
            device=self.device,
        )

    def apply(self, x, y):
        if x.shape != (self.width, self.n) or y.shape != x.shape:
            raise ValueError("coarse preconditioner requires its configured block width")
        if any(v.dtype != wp.float64 or v.device != self.device for v in (x, y)):
            raise ValueError("coarse preconditioner requires FP64 arrays on its device")
        gram(x, self.c, self.coeff, chunk=4096)
        wp.copy(self.work, x)
        if self.balanced:
            self._subtract(self.d, self.coeff, self.work)
        self.smoother.apply(self.work, y)
        if self.balanced:
            gram(y, self.d, self.second, chunk=4096)
        wp.launch(
            _subtract_coefficients,
            self.coeff.shape,
            [self.coeff, self.second, self.balanced],
            device=self.device,
        )
        self._subtract(self.c, self.coeff, y)
        wp.launch(_reject_invalid, y.shape, [y, self.status], device=self.device)


class CoarseCG:
    """Upstream batched CG with a fixed-linear two-level inner preconditioner.

    The coarse basis/factors must stay fixed during each CG solve. Refresh the
    correction before starting the captured outer solve, not inside CG.
    """

    def __init__(self, matrix, correction, width, *, maxiter=25, tol=0.01):
        self.n, self.width, self.device = matrix.shape[0], width, matrix.device
        if correction.n != self.n or correction.width != width or matrix.scalar_type != wp.float64:
            raise ValueError("matching FP64 matrix and coarse-correction width required")
        self.correction = correction
        self.offsets = wp.array(
            [j * self.n for j in range(width + 1)], dtype=wp.int32, device=self.device
        )
        self.operator = RepeatedOperator(matrix, width, self.offsets, "columns")
        self.work = wp.empty((width, self.n), dtype=wp.float64, device=self.device)
        self.linear = linear.LinearOperator(
            (width * self.n, width * self.n),
            wp.float64,
            self.device,
            self.matvec,
            batch_offsets=self.offsets,
            max_batch_length=self.n,
        )
        self.b = wp.zeros(width * self.n, dtype=wp.float64, device=self.device)
        self.x = wp.zeros_like(self.b)
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.state = linear.cg(
            self.operator.linear,
            self.b,
            self.x,
            M=self.linear,
            tol=tol,
            atol=0.0,  # Eigen-residual RHS norms shrink; default atol=tol would stall refinement.
            maxiter=maxiter,
            check_every=0,
            run=False,
        )
        self.requires_static_outer_capture = not _cg_preallocates_loop_control(self.state)

    def matvec(self, x, y, z, alpha, beta):
        self.correction.apply(x.reshape((self.width, self.n)), self.work)
        wp.launch(
            axpby,
            z.size,
            [self.work.flatten(), y, z, wp.float64(alpha), wp.float64(beta)],
            device=self.device,
        )

    def reset_stats(self):
        self.stats.zero_()

    def apply(self, x, y):
        if x.shape != (self.width, self.n) or y.shape != x.shape:
            raise ValueError("CG requires its configured block width")
        if any(v.dtype != wp.float64 or v.device != self.device for v in (x, y)):
            raise ValueError("CG requires FP64 arrays on its device")
        wp.copy(self.b, x.flatten())
        self.x.zero_()
        iterations, residual_sq, tolerance_sq = self.state()
        wp.copy(y.flatten(), self.x)
        wp.launch(
            accumulate_statistics,
            self.width,
            [iterations, residual_sq, tolerance_sq, self.stats],
            device=self.device,
        )
