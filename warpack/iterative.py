"""Device-resident adapters for upstream Warp iterative linear solvers."""

import math
from operator import index

import warp as wp
from warp.optim import linear

from .generalized import _cg_preallocates_loop_control, accumulate_cg_statistics
from .solver import SparseOperator


@wp.kernel
def residual_norms(ax: wp.array[wp.float64], b: wp.array[wp.float64], norms: wp.array[wp.float64]):
    block = wp.tid()
    av = wp.tile_load(ax, shape=256, offset=block * 256)
    bv = wp.tile_load(b, shape=256, offset=block * 256)
    r = av - bv
    wp.tile_atomic_add(norms, wp.tile_sum(wp.tile_map(wp.mul, r, r)), offset=0)
    wp.tile_atomic_add(norms, wp.tile_sum(wp.tile_map(wp.mul, bv, bv)), offset=1)


@wp.kernel
def residual_maxima(norms: wp.array[wp.float64], maxima: wp.array[wp.float64]):
    absolute = wp.sqrt(norms[0])
    relative = wp.float64(0.0)
    if norms[1] > wp.float64(0.0):
        relative = absolute / wp.sqrt(norms[1])
    elif absolute > wp.float64(0.0):
        relative = wp.float64(wp.inf)
    if not wp.isfinite(absolute):
        absolute = wp.float64(wp.inf)
        relative = wp.float64(wp.inf)
    wp.atomic_max(maxima, 0, absolute)
    wp.atomic_max(maxima, 1, relative)


class GMRESInverse:
    """Apply a real FP64 sparse inverse using unmodified upstream Warp GMRES.

    Supports indefinite/nonsymmetric matrices, including a supplied ``K-sigma M``.
    A fixed linear right preconditioner may be supplied as a Warp linear operator.
    It need not be SPD; do not construct an SPD factorization of an indefinite
    matrix or pass a variable inner iterative solve as this preconditioner.

    Sparse topology and addresses are fixed during replay; matrix coefficients
    may change in place. Preconditioner updates remain the caller's responsibility.
    Every RHS starts at zero. Restarted GMRES may exceed ``maxiter`` by up to
    ``restart-1`` iterations; counters report actual upstream iterations.

    ``stats`` contains total iterations, maximum iterations per RHS, failed RHSs,
    and total RHSs. Failures use a fresh original linear residual against the
    upstream effective tolerance (including its dtype safety floor).
    ``residual_maxima`` contains the maximum absolute and relative fresh residual
    since ``reset_stats()``. These diagnostics do not replace outer eigenpair
    residual, orthogonality and requested-spectrum validation.

    Default capture uses a fixed outer budget. ``adaptive_capture=True`` requires
    upstream preallocated loop control (tested on Warp 1.18, absent in 1.15).
    The solver itself, stopping decisions and residual checks execute on device.
    """

    def __init__(
        self,
        matrix,
        tol=1e-12,
        maxiter=1000,
        restart=64,
        preconditioner=None,
        atol=0.0,
        adaptive_capture=False,
    ):
        for name, value in (("tol", tol), ("atol", atol)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if max(tol, atol) == 0:
            raise ValueError("at least one tolerance must be positive")
        for name, value in (("maxiter", maxiter), ("restart", restart)):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a positive integer")
            if index(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.operator = SparseOperator(matrix)
        self.matrix = matrix
        self.n, self.device = self.operator.n, self.operator.device
        self.b = wp.zeros((1, self.n), dtype=wp.float64, device=self.device)
        self.x = wp.zeros_like(self.b)
        self.ax = wp.empty_like(self.b)
        self.norms = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.residual_maxima = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.preconditioner = preconditioner
        self.state = linear.gmres(
            matrix,
            self.b[0],
            self.x[0],
            M=preconditioner,
            tol=tol,
            atol=atol,
            restart=min(index(restart), self.n),
            maxiter=index(maxiter),
            check_every=0,
            run=False,
        )
        if adaptive_capture and not _cg_preallocates_loop_control(self.state):
            raise ValueError(
                "adaptive_capture requires upstream GMRES with preallocated loop control"
            )
        self.requires_static_outer_capture = not adaptive_capture

    def reset_stats(self):
        self.stats.zero_()
        self.residual_maxima.zero_()

    def apply(self, x, y):
        if x.shape != y.shape or len(x.shape) != 2 or x.shape[1] != self.n:
            raise ValueError("expected matching (RHS count, n) arrays")
        if x.dtype != wp.float64 or y.dtype != wp.float64:
            raise TypeError("expected FP64 RHS and output arrays")
        if x.device != self.device or y.device != self.device:
            raise ValueError("RHS and output must share the matrix device")
        for j in range(x.shape[0]):
            wp.copy(self.b, x[j : j + 1])
            self.x.zero_()
            iterations, _, tolerance_sq = self.state(b=self.b[0], x=self.x[0])
            self.operator.apply(self.x, self.ax)
            self.norms.zero_()
            wp.launch_tiled(
                residual_norms,
                dim=(self.n + 255) // 256,
                inputs=[self.ax[0], self.b[0], self.norms],
                block_dim=64,
                device=self.device,
            )
            wp.launch(
                accumulate_cg_statistics,
                1,
                [iterations, self.norms, tolerance_sq, self.stats],
                device=self.device,
            )
            wp.launch(residual_maxima, 1, [self.norms, self.residual_maxima], device=self.device)
            wp.copy(y[j : j + 1], self.x)
