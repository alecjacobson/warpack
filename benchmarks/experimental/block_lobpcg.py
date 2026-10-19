"""Experimental pure-Warp LOBPCG; not part of the supported public API.

Fixed-width rank-revealing canonical orthogonalization keeps zero/dependent
search directions out of Rayleigh--Ritz without host rank decisions. Two metric
orthogonalization passes are deliberately conservative. Performance work follows
independent original-pencil and requested-subspace validation.
"""

import math

import warp as wp

from warpack import kernels as K
from warpack import linalg_tiles as T
from warpack.generalized import generalized_residual
from warpack.solver import SymmetricEigensolver


def gram(x, y, out, chunk=256, symmetric=False):
    from .block_kernels import gram_kernel

    if symmetric:
        from .block_updates import symmetric_gram_kernel

        kernel = symmetric_gram_kernel(chunk)
    else:
        kernel = T.gram_tiled if chunk == 256 else gram_kernel(chunk)
    out.zero_()
    wp.launch_tiled(
        kernel,
        dim=((x.shape[0] + 7) // 8, (y.shape[0] + 7) // 8, (x.shape[1] + chunk - 1) // chunk),
        inputs=[x, y, out],
        block_dim=128,
        device=x.device,
    )


def rotate(x, coeff, out):
    wp.launch_tiled(
        T.rotate_tiled,
        dim=((out.shape[0] + 3) // 4, (out.shape[1] + 31) // 32),
        inputs=[x, coeff, out],
        block_dim=128,
        device=x.device,
    )


@wp.kernel
def scales(g: wp.array2d[wp.float64], scale: wp.array[wp.float64], status: wp.array[wp.int32]):
    i = wp.tid()
    a = g[i, i]
    scale[i] = wp.float64(0.0)
    if not wp.isfinite(a) or a < wp.float64(0.0):
        wp.atomic_max(status, 0, 1)
    elif a > wp.float64(0.0):
        scale[i] = wp.float64(1.0) / wp.sqrt(a)


@wp.kernel
def scale_rows(x: wp.array2d[wp.float64], scale: wp.array[wp.float64]):
    i, j = wp.tid()
    x[i, j] *= scale[i]


@wp.kernel
def normalize_gram(g: wp.array2d[wp.float64], scale: wp.array[wp.float64]):
    i, j = wp.tid()
    if i <= j:
        v = (g[i, j] + g[j, i]) * wp.float64(0.5) * scale[i] * scale[j]
        g[i, j] = v
        g[j, i] = v


@wp.kernel
def symmetrize(g: wp.array2d[wp.float64]):
    i, j = wp.tid()
    if i < j:
        v = (g[i, j] + g[j, i]) * wp.float64(0.5)
        g[i, j] = v
        g[j, i] = v


@wp.kernel
def whitening(
    v: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    order: wp.array[wp.int32],
    coeff: wp.array2d[wp.float64],
    active: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    status: wp.array[wp.int32],
    threshold: wp.float64,
):
    i, j = wp.tid()
    lam = values[order[i]]
    valid = wp.isfinite(lam) and lam > threshold * values[order[0]]
    coeff[i, j] = wp.float64(0.0)
    if valid:
        coeff[i, j] = v[j, order[i]] / wp.sqrt(lam)
    if j == 0:
        active[i] = int(valid)
        if valid:
            wp.atomic_add(rank, 0, 1)
        if not wp.isfinite(lam):
            wp.atomic_max(status, 0, 1)


@wp.kernel
def mask_projected(
    g: wp.array2d[wp.float64],
    active: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    status: wp.array[wp.int32],
    required: int,
):
    # Single thread avoids reading a matrix while other lanes replace entries.
    bound = wp.float64(0.0)
    for i in range(g.shape[0]):
        row = wp.float64(0.0)
        for j in range(g.shape[1]):
            row += wp.abs(g[i, j])
        bound = wp.max(bound, row)
    bound = wp.float64(1.0) + wp.float64(2.0) * bound
    for i in range(g.shape[0]):
        for j in range(i, g.shape[1]):
            v = (g[i, j] + g[j, i]) * wp.float64(0.5)
            if active[i] == 0 or active[j] == 0:
                v = wp.float64(0.0)
                if i == j:
                    v = bound
            g[i, j] = v
            g[j, i] = v
    if rank[0] < required or not wp.isfinite(bound):
        status[0] = 1


@wp.kernel
def discard_x(coeff: wp.array2d[wp.float64], width: int):
    i, j = wp.tid()
    if j < width:
        coeff[i, j] = wp.float64(0.0)


@wp.kernel
def activate_x(active: wp.array[wp.int32], rank: wp.array[wp.int32]):
    i = wp.tid()
    active[i] = 1
    if i == 0:
        rank[0] += active.shape[0]


@wp.kernel
def residual_block(
    ax: wp.array2d[wp.float64],
    bx: wp.array2d[wp.float64],
    vals: wp.array[wp.float64],
    out: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    out[i, j] = ax[i, j] - vals[i] * bx[i, j]


@wp.kernel
def evaluate(
    norms: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    errors: wp.array[wp.float64],
    count: wp.array[wp.int32],
    status: wp.array[wp.int32],
    k: int,
    tol: wp.float64,
):
    i = wp.tid()
    err = wp.sqrt(norms[i, 0]) / wp.max(
        wp.float64(1.0), wp.sqrt(norms[i, 1]) + wp.abs(values[i]) * wp.sqrt(norms[i, 2])
    )
    errors[i] = err
    if not wp.isfinite(err):
        wp.atomic_max(status, 0, 1)
    if i < k and err <= tol and status[0] == 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def copy_values(values: wp.array[wp.float64], order: wp.array[wp.int32], out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = values[order[i]]


class Identity:
    def __init__(self, n, device):
        self.n, self.device = n, device

    def apply(self, x, y):
        wp.copy(y, x)


class UpstreamPreconditioner:
    """Use Warp's own preconditioner application, including FSAI/block Jacobi."""

    def __init__(self, preconditioner):
        self.preconditioner = preconditioner

    def apply(self, x, y):
        for i in range(x.shape[0]):
            self.preconditioner.matvec(x[i], y[i], y[i], 1.0, 0.0)


class ProjectedSolver(SymmetricEigensolver):
    def _diagonalize_projected(self):
        # Prototype-only extension: the 72x72 two-tile kernel fits an L40 block,
        # while 96x96 does not. Keep the production dispatch unchanged.
        if self.m == 72 and self.device.max_shared_memory_per_block >= 98304:
            from warpack.projected import jacobi_kernel

            wp.launch_tiled(
                jacobi_kernel(self.m),
                dim=1,
                inputs=[self.g, self.v],
                block_dim=256,
                device=self.device,
            )
            wp.launch(
                K.sort_diagonal,
                self.m,
                [self.g, self.ritz, self.order, self.which],
                device=self.device,
            )
        else:
            super()._diagonalize_projected()


class BlockLOBPCG:
    """Smallest symmetric eigenpairs with SPD mass and a supplied preconditioner.

    Constraints, if present, must be M-orthonormal rows spanning an invariant
    space to exclude (e.g. rigid motions). This prototype does not implement
    general constrained/KKT eigenproblems. All buffers, including the three-
    block search space, are allocated before capture.
    """

    def __init__(
        self,
        operator,
        k,
        width=None,
        mass=None,
        preconditioner=None,
        initial_vectors=None,
        constraints=None,
        tol=1e-8,
        rank_tol=1e-12,
        gram_chunk=256,
        symmetric_products=False,
        fused_updates=False,
        paired_rotation=False,
    ):
        self.op, self.n, self.device = operator, operator.n, operator.device
        self.k, self.b = k, width or k
        if not 1 <= k <= self.b or 3 * self.b > self.n:
            raise ValueError("require 1 <= k <= width and 3*width <= n")
        self.m = 3 * self.b
        self.mass = mass or Identity(self.n, self.device)
        self.preconditioner = preconditioner or Identity(self.n, self.device)
        self.initial_vectors, self.constraints = initial_vectors, constraints
        self.tol, self.rank_tol = tol, rank_tol
        if gram_chunk < 256 or gram_chunk % 256:
            raise ValueError("gram_chunk must be a positive multiple of 256")
        self.gram_chunk = gram_chunk
        self.symmetric_products = symmetric_products
        self.fused_updates = fused_updates
        self.paired_rotation = paired_rotation
        if not math.isfinite(tol) or tol <= 0 or not 0 < rank_tol < 1:
            raise ValueError("require finite tol > 0 and 0 < rank_tol < 1")
        if self.mass.n != self.n or self.mass.device != self.device:
            raise ValueError("mass must match operator shape and device")
        if constraints is not None and (
            constraints.dtype != wp.float64
            or constraints.ndim != 2
            or constraints.shape[1] != self.n
            or constraints.device != self.device
            or constraints.shape[0] + self.b > self.n
        ):
            raise ValueError("constraints must be device FP64 rows with room for width modes")

        def empty(shape, dtype=wp.float64):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.x, self.ax, self.bx, self.r, self.p = [empty((self.b, self.n)) for _ in range(5)]
        self.s, self.z, self.az = [empty((self.m, self.n)) for _ in range(3)]
        self.u1, self.u2, self.bu = [empty((2 * self.b, self.n)) for _ in range(3)]
        self.sx, self.sw, self.sp = (
            self.s[: self.b],
            self.s[self.b : 2 * self.b],
            self.s[2 * self.b :],
        )
        self.small = ProjectedSolver(Identity(self.m, self.device), self.b, ncv=self.m)
        self.metric = ProjectedSolver(Identity(2 * self.b, self.device), self.b, ncv=2 * self.b)
        self.su = self.s[self.b :]
        self.transform = empty((2 * self.b, 2 * self.b))
        self.coeff = empty((self.b, self.m))
        self.gx, self.lx = [empty((self.b, self.b)) for _ in range(2)]
        self.projection_x = empty((2 * self.b, self.b))
        self.scale = empty(2 * self.b)
        self.xscale = self.scale[: self.b]
        self.active, self.rank, self.status = (
            empty(self.m, wp.int32),
            empty(1, wp.int32),
            empty(1, wp.int32),
        )
        self.values, self.errors = empty(self.b), empty(self.b)
        self.active_x, self.active_u = self.active[: self.b], self.active[self.b :]
        self.order = empty(self.b, wp.int32)
        self.norms = empty((self.b, 3))
        self.converged, self.iterations, self.running = [empty(1, wp.int32) for _ in range(3)]
        self.eigenvalues, self.residuals, self.eigenvectors = (
            self.values[:k],
            self.errors[:k],
            self.x[:k],
        )
        if constraints is not None:
            self.bconstraints = wp.empty_like(constraints)
            self.projection = empty((self.m, constraints.shape[0]))
            self.projection_u = self.projection[self.b :]
        if initial_vectors is not None and (
            initial_vectors.dtype != wp.float64
            or initial_vectors.shape != self.x.shape
            or initial_vectors.device != self.device
        ):
            raise ValueError("prototype requires a device FP64 (width,n) initial block")

    def launch(self, kernel, shape, args):
        wp.launch(kernel, shape, args, device=self.device)

    def gram(self, x, y, out, symmetric=False):
        gram(x, y, out, chunk=self.gram_chunk, symmetric=symmetric and self.symmetric_products)

    def subtract(self, basis, coeff, target, scratch):
        if self.fused_updates:
            from .block_updates import subtract_projection

            wp.launch_tiled(
                subtract_projection,
                dim=((target.shape[0] + 3) // 4, (self.n + 31) // 32),
                inputs=[basis, coeff, target],
                block_dim=128,
                device=self.device,
            )
        else:
            rotate(basis, coeff, scratch)
            self.launch(K.linear_residual, target.shape, [target, scratch])
            wp.copy(target, scratch)

    def project(self):
        if self.constraints is not None:
            for _ in range(2):
                self.gram(self.s, self.bconstraints, self.projection)
                self.subtract(self.constraints, self.projection, self.s, self.z)

    def orthogonalize_x(self):
        for i in range(2):
            self.mass.apply(self.sx, self.bx)
            self.gram(self.sx, self.bx, self.gx, symmetric=True)
            if i == 0:
                self.launch(scales, self.b, [self.gx, self.xscale, self.status])
                self.launch(scale_rows, self.sx.shape, [self.sx, self.xscale])
                self.launch(normalize_gram, self.gx.shape, [self.gx, self.xscale])
            self.launch(K.cholesky, 1, [self.gx, self.lx, self.status])
            self.launch(K.triangular, self.n, [self.sx, self.lx])
        self.mass.apply(self.sx, self.bx)

    def orthogonalize_search(self, x):
        for _ in range(2):
            self.gram(x, self.bx, self.projection_x)
            self.subtract(self.sx, self.projection_x, x, self.bu)
            if self.constraints is not None:
                self.gram(x, self.bconstraints, self.projection_u)
                self.subtract(self.constraints, self.projection_u, x, self.bu)

    def whiten(self, x, out, normalize=False):
        self.mass.apply(x, self.bu)
        self.gram(x, self.bu, self.metric.g, symmetric=True)
        if normalize:
            self.launch(scales, 2 * self.b, [self.metric.g, self.scale, self.status])
            self.launch(scale_rows, x.shape, [x, self.scale])
            self.launch(normalize_gram, self.metric.g.shape, [self.metric.g, self.scale])
        else:
            self.launch(symmetrize, self.metric.g.shape, [self.metric.g])
        self.metric.which = 1  # Descending metric eigenvalues, independent directions first.
        self.metric.diagonalize()
        self.rank.zero_()
        self.launch(
            whitening,
            self.transform.shape,
            [
                self.metric.v,
                self.metric.ritz,
                self.metric.order,
                self.transform,
                self.active_u,
                self.rank,
                self.status,
                self.rank_tol,
            ],
        )
        rotate(x, self.transform, out)

    def rayleigh_ritz(self):
        self.project()
        self.orthogonalize_x()
        self.orthogonalize_search(self.su)
        self.whiten(self.su, self.u1, normalize=True)
        self.orthogonalize_search(self.u1)
        self.whiten(self.u1, self.u2)
        wp.copy(self.su, self.u2)
        self.launch(activate_x, self.b, [self.active_x, self.rank])
        self.op.apply(self.s, self.az)
        self.gram(self.s, self.az, self.small.g, symmetric=True)
        self.launch(mask_projected, 1, [self.small.g, self.active, self.rank, self.status, self.b])
        self.small.which = 0
        self.small.diagonalize()
        self.launch(
            K.rotation_coefficients, self.coeff.shape, [self.small.v, self.small.order, self.coeff]
        )
        if self.paired_rotation:
            from .block_updates import paired_ritz_rotation

            wp.launch_tiled(
                paired_ritz_rotation,
                dim=((self.b + 3) // 4, (self.n + 31) // 32),
                inputs=[
                    self.sx,
                    self.su,
                    self.coeff[:, : self.b],
                    self.coeff[:, self.b :],
                    self.x,
                    self.p,
                ],
                block_dim=128,
                device=self.device,
            )
        else:
            rotate(self.s, self.coeff, self.x)
            if self.fused_updates:
                # P excludes X; avoid multiplying known zero columns.
                rotate(self.su, self.coeff[:, self.b :], self.p)
            else:
                self.launch(discard_x, self.coeff.shape, [self.coeff, self.b])
                rotate(self.s, self.coeff, self.p)
        self.launch(copy_values, self.b, [self.small.ritz, self.small.order, self.values])
        self.finalize()

    def initialize(self, seed=42):
        self.status.zero_()
        self.iterations.zero_()
        if hasattr(self.preconditioner, "reset_stats"):
            self.preconditioner.reset_stats()
        self.s.zero_()
        if self.constraints is not None:
            self.mass.apply(self.constraints, self.bconstraints)
        if self.initial_vectors is None:
            self.launch(K.random_block, self.sx.shape, [self.sx, seed])
        else:
            wp.copy(self.sx, self.initial_vectors)
        self.launch(K.identity_order, self.b, [self.order])
        self.rayleigh_ritz()

    def step(self):
        wp.copy(self.sx, self.x)
        self.preconditioner.apply(self.r, self.sw)
        wp.copy(self.sp, self.p)
        self.rayleigh_ritz()

    def finalize(self):
        self.op.apply(self.x, self.ax)
        self.mass.apply(self.x, self.bx)
        self.launch(residual_block, self.r.shape, [self.ax, self.bx, self.values, self.r])
        self.norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.b, (self.n + 255) // 256),
            inputs=[self.x, self.ax, self.bx, self.values, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        self.launch(
            evaluate,
            self.b,
            [self.norms, self.values, self.errors, self.converged, self.status, self.k, self.tol],
        )

    def solve(self, steps=100):
        self.initialize()
        for _ in range(steps):
            self.step()
        return self

    def capture(self, max_iterations=100, adaptive=None):
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        static_only = getattr(self.preconditioner, "requires_static_outer_capture", False)
        if adaptive is None:
            adaptive = not static_only
        if adaptive and static_only:
            raise ValueError("This preconditioner requires adaptive=False for the outer graph")
        self.solve(1)
        wp.synchronize_device(self.device)

        def body():
            self.step()
            self.launch(
                K.loop_update,
                1,
                [
                    self.converged,
                    self.status,
                    self.iterations,
                    self.running,
                    self.k,
                    max_iterations,
                ],
            )

        with wp.ScopedCapture(device=self.device) as capture:
            self.initialize()
            self.running.fill_(1)
            if adaptive:
                wp.capture_while(self.running, body)
            else:
                for _ in range(max_iterations):
                    self.step()
                self.iterations.fill_(max_iterations)
        self.graph = capture.graph
        return self.graph
