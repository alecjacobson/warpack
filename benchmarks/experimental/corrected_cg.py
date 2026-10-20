"""Experimental original-residual correction around unmodified upstream CG.

One RHS per apply. The initial solve uses the requested relative tolerance;
each correction solves the fresh residual to a separately declared tolerance.
This does not change upstream CG or guarantee an unattainable residual floor.
"""

import warp as wp

from warpack import CGInverse, SparseOperator
from warpack.iterative import residual_maxima, residual_norms
from warpack.kernels import add_correction, linear_residual


@wp.kernel
def remember(base: wp.array[wp.int32], correction: wp.array[wp.int32], old: wp.array[wp.int32]):
    old[0] = base[0] + correction[0]
    old[1] = base[2] + correction[2]


@wp.kernel
def record(
    base: wp.array[wp.int32],
    correction: wp.array[wp.int32],
    old: wp.array[wp.int32],
    norms: wp.array[wp.float64],
    tol: wp.float64,
    recursive: wp.array[wp.int32],
    fresh: wp.array[wp.int32],
):
    iterations = base[0] + correction[0] - old[0]
    recursive[0] += iterations
    recursive[1] = wp.max(recursive[1], iterations)
    recursive[2] += int(base[2] + correction[2] > old[1])
    recursive[3] += 1
    fresh[0] += iterations
    fresh[1] = wp.max(fresh[1], iterations)
    fresh[2] += int(not wp.isfinite(norms[0]) or norms[0] > tol * tol * norms[1])
    fresh[3] += 1


class CorrectedCGInverse:
    """Fixed extra upstream CG solves; original RHS remains the audit reference.

    Both stats arrays count original RHS calls and total initial+correction
    iterations. Recursive failures mean any constituent CG solve failed its own
    stopping check; fresh failures check the final original equation at ``tol``.
    The declared experimental relative audit has no absolute tolerance. It is
    separate from the native per-state dtype safety floors. The two native
    state counters remain accessible as ``base.stats`` and ``correction.stats``.
    """

    requires_static_outer_capture = True
    width = 1

    def __init__(
        self,
        matrix,
        tol=1e-11,
        correction_tol=0.01,
        correction_steps=1,
        maxiter=10000,
        preconditioner=None,
    ):
        if correction_steps < 0 or int(correction_steps) != correction_steps:
            raise ValueError("correction_steps must be a nonnegative integer")
        if not 1e-13 <= tol < 1 or not 0 < correction_tol < 1:
            raise ValueError("experimental audit requires 1e-13<=tol<1 and 0<correction_tol<1")
        self.base = CGInverse(
            matrix, tol=tol, atol=0.0, maxiter=maxiter, preconditioner=preconditioner
        )
        self.correction = CGInverse(
            matrix,
            tol=correction_tol,
            atol=0.0,
            maxiter=maxiter,
            preconditioner=self.base.preconditioner,
        )
        self.operator = SparseOperator(matrix)
        self.n, self.device = self.base.n, self.base.device
        self.tol, self.correction_steps = tol, int(correction_steps)
        self.rhs = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        self.ax = wp.empty_like(self.rhs)
        self.delta = wp.empty_like(self.rhs)
        self.norms = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.previous = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.fresh_stats = wp.zeros_like(self.stats)
        self.residual_maxima = wp.zeros(2, dtype=wp.float64, device=self.device)

    def reset_stats(self):
        self.base.reset_stats()
        self.correction.reset_stats()
        self.stats.zero_()
        self.fresh_stats.zero_()
        self.residual_maxima.zero_()

    def apply(self, x, y):
        if x.shape != (1, self.n) or y.shape != x.shape:
            raise ValueError("experimental correction requires one RHS")
        if any(
            a.dtype != wp.float64 or a.device != self.device or a.strides[1] != 8 for a in (x, y)
        ):
            raise ValueError("expected contiguous FP64 rows on the operator device")
        for workspace in (self.rhs, self.ax, self.delta):
            if max(y.ptr, workspace.ptr) < min(y.ptr + 8 * self.n, workspace.ptr + 8 * self.n):
                raise ValueError("output cannot overlap correction workspaces")
        wp.copy(self.rhs, x)
        wp.launch(
            remember, 1, [self.base.stats, self.correction.stats, self.previous], device=self.device
        )
        self.base.apply(self.rhs, y)
        for _ in range(self.correction_steps):
            self.operator.apply(y, self.ax)
            wp.launch(linear_residual, y.shape, [self.rhs, self.ax], device=self.device)
            self.correction.apply(self.ax, self.delta)
            wp.launch(add_correction, y.shape, [y, self.delta], device=self.device)
        self.operator.apply(y, self.ax)
        self.norms.zero_()
        wp.launch_tiled(
            residual_norms,
            dim=(self.n + 255) // 256,
            inputs=[self.ax[0], self.rhs[0], self.norms],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            record,
            1,
            [
                self.base.stats,
                self.correction.stats,
                self.previous,
                self.norms,
                self.tol,
                self.stats,
                self.fresh_stats,
            ],
            device=self.device,
        )
        wp.launch(residual_maxima, 1, [self.norms, self.residual_maxima], device=self.device)
