"""Restarted Krylov iteration in the SPD mass inner product."""

import warp as wp

from . import kernels as K
from . import krylov_kernels as KR
from .generalized import generalized_outputs, generalized_residual
from .solver import KrylovSchur


@wp.kernel
def metric_dots(
    q: wp.array2d[wp.float64],
    mass_vector: wp.array2d[wp.float64],
    parts: wp.array2d[wp.float64],
):
    r, chunk = wp.tid()
    a = wp.tile_load(q[r], shape=1024, offset=chunk * 1024)
    b = wp.tile_load(mass_vector[0], shape=1024, offset=chunk * 1024)
    wp.tile_store(parts[r], wp.tile_sum(wp.tile_map(wp.mul, a, b)), offset=chunk)


@wp.kernel
def metric_norm(
    q: wp.array2d[wp.float64],
    mass_vector: wp.array2d[wp.float64],
    parts: wp.array[wp.float64],
    j: int,
):
    chunk = wp.tid()
    a = wp.tile_load(q[j], shape=1024, offset=chunk * 1024)
    b = wp.tile_load(mass_vector[0], shape=1024, offset=chunk * 1024)
    wp.tile_store(parts, wp.tile_sum(wp.tile_map(wp.mul, a, b)), offset=chunk)


@wp.kernel
def quotient_parts(
    x: wp.array2d[wp.float64],
    ax: wp.array2d[wp.float64],
    bx: wp.array2d[wp.float64],
    sums: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    v = wp.tile_load(x[j], shape=256, offset=chunk * 256)
    a = wp.tile_load(ax[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(bx[j], shape=256, offset=chunk * 256)
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, v, a)), offset=0)
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, v, b)), offset=1)


@wp.kernel
def quotients(sums: wp.array2d[wp.float64], values: wp.array[wp.float64]):
    j = wp.tid()
    values[j] = sums[j, 0] / sums[j, 1]


@wp.kernel
def add_scaled(x: wp.array2d[wp.float64], y: wp.array2d[wp.float64], scale: wp.float64):
    i, j = wp.tid()
    x[i, j] += scale * y[i, j]


class _CayleyRHS:
    def __init__(self, operator, mass, sigma):
        self.operator, self.mass, self.sigma = operator, mass, float(sigma)
        self.n, self.device = operator.n, operator.device
        self.mx = wp.empty((1, self.n), dtype=wp.float64, device=self.device)

    def apply(self, x, y):
        self.operator.apply(x, y)
        self.mass.apply(x, self.mx)
        wp.launch(add_scaled, y.shape, [y, self.mx, self.sigma], device=self.device)


class _Composition:
    def __init__(self, rhs_operator, inverse):
        self.rhs_operator, self.inverse = rhs_operator, inverse
        self.n, self.device = rhs_operator.n, rhs_operator.device
        self.requires_static_outer_capture = getattr(
            inverse, "requires_static_outer_capture", False
        )
        self.rhs = wp.empty((1, self.n), dtype=wp.float64, device=self.device)

    def apply(self, x, y):
        for j in range(x.shape[0]):
            self.rhs_operator.apply(x[j : j + 1], self.rhs)
            self.inverse.apply(self.rhs, y[j : j + 1])


class GeneralizedKrylovSchur(KrylovSchur):
    """Real symmetric A x = lambda M x, with SPD M, using mass-orthogonal Krylov.

    In ``mode="shift-invert"``, supply a one-RHS inverse of A-target*M;
    the requested pairs are nearest ``target`` (LM on the inverse operator).
    In ``mode="regular"``, supply M^-1 and select physical LM/LA/SA with
    ``which``. M^-1 A is self-adjoint in the M metric, not Euclidean space.
    ``mode="cayley"`` uses (A-target*M)^-1 (A+target*M), with nonzero target,
    and ranks by |(lambda+target)/(lambda-target)|. Supply the same shifted
    inverse as in shift-invert mode; output remains in the original pencil.

    All products, orthogonalization, Ritz solves, Rayleigh quotients and physical
    residuals run in Warp, apart from the optional cuDSS inverse. The mass and
    stiffness operators must be real symmetric; positive definiteness of M is
    a caller precondition. Repeated eigenvalues need sufficiently rich starting
    spaces/restarts; residual convergence alone is not a completeness proof.
    Default capture uses a fixed outer budget if the original operator, mass,
    or inverse advertises ``requires_static_outer_capture``. Explicit adaptive
    capture rejects such compositions before warmup or graph construction.
    """

    def __init__(
        self,
        operator,
        mass,
        inverse,
        k,
        ncv=None,
        tol=1e-9,
        target=0.0,
        mode="shift-invert",
        which="LM",
        initial_vector=None,
    ):
        if mode not in ("shift-invert", "regular", "cayley"):
            raise ValueError("mode must be shift-invert, regular or cayley")
        if mode in ("shift-invert", "cayley") and which != "LM":
            raise ValueError("transformed modes support largest transformed magnitude (LM) only")
        if mode == "cayley" and target == 0:
            raise ValueError("Cayley target must be nonzero")
        if mode == "regular" and which not in ("LM", "LA", "SA"):
            raise ValueError("regular mode supports LM, LA or SA")
        if any(op.n != operator.n or op.device != operator.device for op in (mass, inverse)):
            raise ValueError("operators must have matching shape and device")
        if getattr(inverse, "width", 1) != 1:
            raise ValueError("generalized Krylov requires a one-RHS inverse")
        self.original, self.mass = operator, mass
        self.shift, self.mode = target, mode
        rhs_operator = mass if mode == "shift-invert" else operator
        if mode == "cayley":
            rhs_operator = _CayleyRHS(operator, mass, target)
        iteration = _Composition(rhs_operator, inverse)
        # The original operator also participates in finalization, and mass
        # products in metric orthogonalization, beyond the iteration product.
        # Preserve every operand's restriction, including implicit operators
        # that contain sparse solves and the two operands of the Cayley RHS.
        iteration.requires_static_outer_capture = any(
            getattr(op, "requires_static_outer_capture", False) for op in (operator, mass, inverse)
        )
        super().__init__(iteration, k, ncv=ncv, which=which, tol=tol, initial_vector=initial_vector)
        self.metric_vector = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        self.bx = wp.empty_like(self.eigenvectors)
        self.physical_values = wp.empty_like(self.eigenvalues)
        self.physical_order = wp.empty(self.k, dtype=wp.int32, device=self.device)
        self.sums = wp.zeros((self.k, 2), dtype=wp.float64, device=self.device)
        self.general_norms = wp.zeros((self.k, 3), dtype=wp.float64, device=self.device)

    def _metric_norm(self, j, check=False):
        self.mass.apply(self.views[j], self.metric_vector)
        wp.launch_tiled(
            metric_norm,
            dim=self.norm_parts.size,
            inputs=[self.q, self.metric_vector, self.norm_parts, j],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            KR.finish_norm,
            dim=1,
            inputs=[
                self.norm_parts,
                self.norm,
                self.source_norm,
                self.breakdown,
                self.recoveries,
                check,
            ],
            block_dim=128,
            device=self.device,
        )

    def _orthogonalize_column(self, j, check):
        for pass_index in range(2):
            self.mass.apply(self.views[j], self.metric_vector)
            rows = j + int(pass_index == 0)
            wp.launch_tiled(
                metric_dots,
                dim=(rows, self.dot_parts.shape[1]),
                inputs=[self.q, self.metric_vector, self.dot_parts],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                KR.finish_dots,
                dim=rows,
                inputs=[
                    self.dot_parts,
                    self.dots,
                    self.source_norm,
                    self.g,
                    j,
                    check and pass_index == 0,
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                KR.subtract,
                dim=self.norm_parts.size,
                inputs=[self.q, self.dots, self.norm_parts, j, False],
                block_dim=128,
                device=self.device,
            )
        self._metric_norm(j, check)

    def _recover_eager(self, j):
        wp.launch(
            K.fallback_seed,
            self.n,
            [self.q, self.breakdown, self.recoveries, j],
            device=self.device,
        )
        self._orthogonalize_column(j, False)

    def initialize(self, seed=42):
        self.recoveries.zero_()
        self.g.zero_()
        self.status.zero_()
        if self.initial_vector is None:
            wp.launch(K.random_block, (1, self.n), [self.q, seed], device=self.device)
        else:
            wp.copy(self.views[0], self.initial_vector)
        self._metric_norm(0)
        wp.launch(K.basis_scale, self.n, [self.q, self.norm, 0], device=self.device)
        for j in range(1, self.keep):
            self.expand(j)

    def complete_projection(self):
        self.mass.apply(self.zviews[-1], self.metric_vector)
        wp.launch_tiled(
            KR.last_projection,
            dim=(self.m, self.dot_parts.shape[1]),
            inputs=[self.q, self.metric_vector, self.dot_parts],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            KR.finish_projection,
            dim=self.m,
            inputs=[self.dot_parts, self.g],
            block_dim=128,
            device=self.device,
        )

    def finalize(self):
        wp.copy(self.eigenvectors, self.retained, count=self.k * self.n)
        self.original.apply(self.eigenvectors, self.ax)
        self.mass.apply(self.eigenvectors, self.bx)
        self.sums.zero_()
        wp.launch_tiled(
            quotient_parts,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.bx, self.sums],
            block_dim=128,
            device=self.device,
        )
        wp.launch(quotients, self.k, [self.sums, self.physical_values], device=self.device)
        wp.launch(K.identity_order, self.k, [self.physical_order], device=self.device)
        self.general_norms.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[
                self.eigenvectors,
                self.ax,
                self.bx,
                self.physical_values,
                self.physical_order,
                self.general_norms,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            generalized_outputs,
            self.k,
            [
                self.physical_values,
                self.physical_order,
                self.general_norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )
