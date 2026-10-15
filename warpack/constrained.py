"""Finite modes of symmetric equality-constrained pencils, in physical coordinates."""

import math

import warp as wp
import warp.sparse as ws

from .generalized_krylov import GeneralizedKrylovSchur, quotient_parts, quotients


def _scalar_csr(matrix):
    if matrix.values.dtype != wp.float64 or matrix.row_counts is not None:
        raise ValueError("expected compressed scalar FP64 BSR")


@wp.kernel
def _triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    vals: wp.array[wp.float64],
    start: int,
    scale: wp.float64,
    n: int,
    constraint: bool,
):
    i = wp.tid()
    for p in range(offsets[i], offsets[i + 1]):
        if constraint:
            rows[start + 2 * p] = n + i
            cols[start + 2 * p] = columns[p]
            vals[start + 2 * p] = values[p]
            rows[start + 2 * p + 1] = columns[p]
            cols[start + 2 * p + 1] = n + i
            vals[start + 2 * p + 1] = values[p]
        else:
            rows[start + p] = i
            cols[start + p] = columns[p]
            vals[start + p] = scale * values[p]


def saddle_point_matrix(stiffness, mass, constraints, shift=0.0):
    """Assemble [[H-shift*M, C.T], [C, 0]] using Warp sparse kernels.

    Inputs are full compressed scalar FP64 BSR matrices, with H/M n-by-n and
    C r-by-n. This is setup (sparse structure/count synchronization is allowed),
    not a captured numeric update. Refactor after changing matrices or shift.
    C must have independent rows; H-shift*M must be nonsingular on ker(C).
    """
    n = stiffness.shape[0]
    if stiffness.shape != (n, n) or mass.shape != (n, n):
        raise ValueError("stiffness and mass must be square and have matching shape")
    if constraints.shape[1] != n or not 0 < constraints.shape[0] < n:
        raise ValueError("constraints must have shape (r, n), with 0 < r < n")
    if not math.isfinite(shift):
        raise ValueError("shift must be finite")
    for a in (stiffness, mass, constraints):
        _scalar_csr(a)
        if a.device != stiffness.device:
            raise ValueError("matrices must be on the same device")
        a.nnz_sync()
    count = stiffness.nnz + mass.nnz + 2 * constraints.nnz
    device = stiffness.device
    rows = wp.zeros(count, dtype=wp.int32, device=device)
    cols = wp.zeros_like(rows)
    vals = wp.zeros(count, dtype=wp.float64, device=device)
    start = 0
    for a, scale, constraint in (
        (stiffness, 1.0, False),
        (mass, -shift, False),
        (constraints, 1.0, True),
    ):
        wp.launch(
            _triplets,
            a.nrow,
            [a.offsets, a.columns, a.values, rows, cols, vals, start, scale, n, constraint],
            device=device,
        )
        start += a.nnz * (2 if constraint else 1)
    size = n + constraints.shape[0]
    return ws.bsr_from_triplets(size, size, rows, cols, vals)


class _PhysicalInverse:
    def __init__(self, inverse, n):
        self.inverse, self.n, self.device = inverse, n, inverse.device
        self.requires_static_outer_capture = getattr(
            inverse, "requires_static_outer_capture", False
        )
        self.rhs = wp.zeros((1, inverse.n), dtype=wp.float64, device=self.device)
        self.solution = wp.empty_like(self.rhs)

    def apply(self, x, y):
        for j in range(x.shape[0]):
            # Reset multiplier RHS too, including after captured replays.
            self.rhs.zero_()
            wp.copy(self.rhs, x[j : j + 1], count=self.n)
            self.inverse.apply(self.rhs, self.solution)
            wp.copy(y[j : j + 1], self.solution, count=self.n)


@wp.kernel
def _rectangular_product(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    j, i = wp.tid()
    value = wp.float64(0.0)
    for p in range(offsets[i], offsets[i + 1]):
        value += values[p] * x[j, columns[p]]
    y[j, i] = value


@wp.kernel
def _long_row_product(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    j, i, lane = wp.tid()
    value = wp.float64(0.0)
    for p in range(offsets[i] + lane, offsets[i + 1], 128):
        value += values[p] * x[j, columns[p]]
    total = wp.tile_sum(wp.tile(value))
    if lane == 0:
        y[j, i] = wp.tile_extract(total, 0)


@wp.kernel
def _multipliers(
    solution: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    mu: wp.array2d[wp.float64],
    n: int,
    j: int,
    shift: wp.float64,
):
    i = wp.tid()
    mu[j, i] = (values[j] - shift) * solution[0, n + i]


@wp.kernel
def _constraint_norms(
    offsets: wp.array[wp.int32], values: wp.array[wp.float64], norms: wp.array[wp.float64]
):
    i = wp.tid()
    value = wp.float64(0.0)
    for p in range(offsets[i], offsets[i + 1]):
        value += values[p] * values[p]
    norms[i] = wp.sqrt(value)


@wp.kernel
def _equilibrium_parts(
    x: wp.array2d[wp.float64],
    h: wp.array2d[wp.float64],
    m: wp.array2d[wp.float64],
    force: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    sums: wp.array2d[wp.float64],
):
    j, chunk = wp.tid()
    a = wp.tile_load(h[j], shape=256, offset=chunk * 256)
    b = wp.tile_load(m[j], shape=256, offset=chunk * 256)
    f = wp.tile_load(force[j], shape=256, offset=chunk * 256)
    v = wp.tile_load(x[j], shape=256, offset=chunk * 256)
    r = a + f - b * values[j]
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, r, r)), offset=0)
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, a, a)), offset=1)
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, b, b)), offset=2)
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, f, f)), offset=3)
    wp.tile_atomic_add(sums[j], wp.tile_sum(wp.tile_map(wp.mul, v, v)), offset=4)


@wp.kernel
def _outputs(
    values: wp.array[wp.float64],
    sums: wp.array2d[wp.float64],
    cw: wp.array2d[wp.float64],
    cnorm: wp.array[wp.float64],
    errors: wp.array[wp.float64],
    constraint_errors: wp.array[wp.float64],
    tol: wp.float64,
    converged: wp.array[wp.int32],
):
    j = wp.tid()
    denom = wp.max(
        wp.float64(1.0),
        wp.sqrt(sums[j, 1]) + wp.abs(values[j]) * wp.sqrt(sums[j, 2]) + wp.sqrt(sums[j, 3]),
    )
    err = wp.sqrt(sums[j, 0]) / denom
    cerr = wp.float64(0.0)
    for i in range(cw.shape[1]):
        d = cnorm[i] * wp.sqrt(sums[j, 4])
        if not wp.isfinite(d) or d <= wp.float64(0.0) or not wp.isfinite(cw[j, i]):
            cerr = wp.float64(wp.inf)
        else:
            cerr = wp.max(cerr, wp.abs(cw[j, i]) / d)
    errors[j] = err
    constraint_errors[j] = cerr
    if wp.isfinite(values[j]) and wp.isfinite(err) and err <= tol and cerr <= tol:
        wp.atomic_add(converged, 0, 1)


class ConstrainedKrylovSchur(GeneralizedKrylovSchur):
    """H W + C.T mu = M W lambda, C W = 0, W.T M W = I.

    H is real symmetric, M is SPD on the n physical coordinates, and C is a
    compressed scalar FP64 r-by-n BSR with independent rows. Supply a one-RHS
    inverse of [[H-target*M, C.T], [C, 0]], e.g. CuDSSInverse with general or
    symmetric-indefinite factors. Factorization must be nonsingular. Dependent
    constraints must be removed by the caller before factorization.

    The iteration is [I,0] KKT^-1 [M;0], self-adjoint in the physical M metric.
    Its nonzero spectrum gives the finite constrained modes nearest target;
    its r zero eigenvalues represent the eliminated infinite spectrum. No
    dense nullspace basis or singular mass inner product is used. Starts and
    breakdown recovery may be infeasible; constraint residuals are explicitly
    checked on output. Residual convergence alone does not prove completeness.

    Outputs include row-major eigenvectors (k,n), multipliers (k,r), equilibrium
    residuals ||H w+C.T mu-lambda M w||/max(1,||H w||+||C.T mu||+|lambda| ||M w||),
    and constraint_residuals max_i |C_i w|/(||C_i|| ||w||). Both must pass tol.
    All numeric iteration and validation run in Warp except the supplied inverse.
    Replay uses preallocated buffers; cuDSS requires a fixed outer cycle budget.
    Owns no factors: release graphs before closing the caller-owned inverse.
    """

    def __init__(
        self,
        operator,
        mass,
        constraints,
        inverse,
        k,
        ncv=None,
        tol=1e-9,
        target=0.0,
        initial_vector=None,
    ):
        _scalar_csr(constraints)
        n, r = operator.n, constraints.shape[0]
        if constraints.shape[1] != n or not 0 < r < n:
            raise ValueError("constraints must have shape (r, n), with 0 < r < n")
        if not 1 <= k < n - r:
            raise ValueError("require 1 <= k < n-r finite modes")
        if (
            inverse.n != n + r
            or inverse.device != operator.device
            or constraints.device != operator.device
        ):
            raise ValueError("KKT inverse and constraints must have matching shape and device")
        if getattr(inverse, "width", 1) != 1:
            raise ValueError("KKT inverse must have one RHS")
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        physical = _PhysicalInverse(inverse, n)
        super().__init__(
            operator,
            mass,
            physical,
            k,
            ncv=ncv,
            tol=tol,
            target=target,
            initial_vector=initial_vector,
        )
        self.physical_inverse = physical
        self.constraints = constraints
        self.constraint_transpose = ws.bsr_transposed(constraints)
        self.multipliers = wp.empty((k, r), dtype=wp.float64, device=self.device)
        self.constraint_values = wp.empty_like(self.multipliers)
        self.constraint_residuals = wp.empty(k, dtype=wp.float64, device=self.device)
        self.constraint_norms = wp.empty(r, dtype=wp.float64, device=self.device)
        self.constraint_force = wp.empty_like(self.eigenvectors)
        self.inverse_image = wp.empty((1, n), dtype=wp.float64, device=self.device)
        self.constrained_sums = wp.zeros((k, 5), dtype=wp.float64, device=self.device)
        wp.launch(
            _constraint_norms,
            r,
            [constraints.offsets, constraints.values, self.constraint_norms],
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
        wp.launch(quotients, self.k, [self.sums, self.eigenvalues], device=self.device)
        # If KKT [z;nu]=[M w;0], an exact eigenpair has mu=(lambda-target)*nu.
        # Always evaluate the ORIGINAL equilibrium and constraints afterward.
        for j in range(self.k):
            self.physical_inverse.apply(self.bx[j : j + 1], self.inverse_image)
            wp.launch(
                _multipliers,
                self.multipliers.shape[1],
                [
                    self.physical_inverse.solution,
                    self.eigenvalues,
                    self.multipliers,
                    self.n,
                    j,
                    self.shift,
                ],
                device=self.device,
            )
        for a, x, y in (
            (self.constraints, self.eigenvectors, self.constraint_values),
            (self.constraint_transpose, self.multipliers, self.constraint_force),
        ):
            if a.nnz / a.nrow > 64:
                wp.launch_tiled(
                    _long_row_product,
                    dim=y.shape,
                    inputs=[a.offsets, a.columns, a.values, x, y],
                    block_dim=128,
                    device=self.device,
                )
            else:
                wp.launch(
                    _rectangular_product,
                    y.shape,
                    [a.offsets, a.columns, a.values, x, y],
                    device=self.device,
                )
        self.constrained_sums.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            _equilibrium_parts,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[
                self.eigenvectors,
                self.ax,
                self.bx,
                self.constraint_force,
                self.eigenvalues,
                self.constrained_sums,
            ],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            _outputs,
            self.k,
            [
                self.eigenvalues,
                self.constrained_sums,
                self.constraint_values,
                self.constraint_norms,
                self.residuals,
                self.constraint_residuals,
                self.tol,
                self.converged,
            ],
            device=self.device,
        )
