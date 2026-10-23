"""Graph-capturable FP64 preconditioned block eigensolving in Warp.

Fixed-width rank-revealing canonical orthogonalization keeps zero/dependent
search directions out of Rayleigh--Ritz without host rank decisions. Two metric
orthogonalization passes are deliberately conservative. Performance work follows
independent original-pencil and requested-subspace validation.
"""

import math
from operator import index as integer_index

import warp as wp

from . import kernels as K
from . import linalg_tiles as T
from .constraints import MassOrthogonalComplementInverse
from .generalized import generalized_residual
from .solver import SymmetricEigensolver


def gram(x, y, out, chunk=256, symmetric=False):
    from .block_products import gram_kernel

    if symmetric:
        from .block_products import symmetric_gram_kernel

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
        self.n, self.device = preconditioner.shape[0], preconditioner.device
        # Upstream shapes count scalar DOFs. BSR preconditioners retain a matrix
        # element dtype, but their matvec accepts scalar vectors directly.
        # Diagonal LinearOperators retain their (n,) diagonal shape.
        if (
            preconditioner.shape not in ((self.n,), (self.n, self.n))
            or getattr(preconditioner, "scalar_type", preconditioner.dtype) != wp.float64
        ):
            raise ValueError("upstream preconditioner must be square FP64")
        self.requires_static_outer_capture = getattr(
            preconditioner, "requires_static_outer_capture", False
        )

    def apply(self, x, y):
        for i in range(x.shape[0]):
            self.preconditioner.matvec(x[i], y[i], y[i], 1.0, 0.0)


class ProjectedSolver(SymmetricEigensolver):
    """Block projection with the measured Ada global path and existing fallback."""

    def _diagonalize_projected(self):
        # Keep the established shared-72 path on other devices and when the
        # bounded global loop is unavailable. Ada uses the common fused path.
        if (
            self.m == 72
            and self.device.max_shared_memory_per_block >= 98304
            and not (self._prefer_global_projected and self._global_jacobi is not None)
        ):
            from .projected import jacobi_kernel

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
    """Smallest algebraic eigenpairs of symmetric K x = lambda M x.

    Operators expose ``n``, ``device`` and ``apply(rows, out)`` for FP64 device
    rows. Mass must be SPD; omitted mass is identity. A supplied preconditioner
    may expose that protocol or be an upstream Warp FP64 LinearOperator/FSAI.
    Matrix-free applications must be device-only and graph capturable.

    ``1 <= k <= width`` and ``3*width <= n`` are required. Initial vectors are
    1..width independent FP64 rows; remaining rows are seeded randomly. Input
    arrays stay caller-owned and unmodified. Output eigenvectors are rows too.
    Constraints span an invariant space to exclude, e.g. rigid modes. By default
    they must be M-orthonormal; ``normalize_constraints=True`` normalizes an
    owned copy with two device CholeskyQR passes. Arbitrary KKT constraints and
    singular mass are outside this contract.

    ``capture()`` warms and synchronizes setup, then records initialization and
    iteration. Replays consume current input-buffer values. Sparse topology and
    captured storage identities must stay fixed; updating preconditioner factors
    remains the caller's responsibility. Instances own scratch and must not be
    used concurrently on independent streams. Inputs and solver must stay live.

    ``status == 0`` and ``converged == k`` report finite original-equation
    residuals below tol, normalized by max(1, ||Kx|| + |lambda| ||Mx||). They are
    not a global requested-spectrum certificate for an arbitrary supplied start.
    In particular, an exact invariant warm block can omit newly lower modes
    after eigenvalue crossings. ``initial_perturbation`` adds seeded random
    exploration relative to each supplied row's Euclidean norm. It is explicit
    and defaults to zero; it improves exploration but is not a completeness
    proof. Rank-deficient/non-finite inputs set nonzero device status, not a
    numerical host exception. Budget exhaustion leaves an incomplete count.

    ``reuse_operator_images`` carries A-images through basis transformations.
    With it enabled, ``search_bucket_size > 0`` selects a preallocated search
    width on device from current residuals; ``keep_history`` retains every
    history row (otherwise only selected rows). ``preconditioner_bucket_size``
    similarly selects RHS batches for a BatchedCGPreconditioner whose width must
    equal the block width. Zero disables bucketing. These options trade setup
    memory and capture time for less replay work; sizes/settings are problem
    dependent. Topology and operator coefficients must remain fixed during each
    solve, but coefficients may change between graph replays.

    The product/rotation tuning switches are opt-in. Existing Krylov defaults
    are unaffected. All numerical work here is FP64 Warp; a supplied inverse
    may independently use upstream CG or optional cuDSS.
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
        normalize_constraints=False,
        initial_perturbation=0.0,
        reuse_operator_images=False,
        search_bucket_size=0,
        keep_history=True,
        preconditioner_bucket_size=0,
        compact_projected_graphs=False,
    ):
        if not all(hasattr(operator, attr) for attr in ("n", "device", "apply")):
            raise ValueError("operator must expose n, device and apply")
        self.op, self.n, self.device = operator, integer_index(operator.n), operator.device
        self.k = k = integer_index(k)
        self.b = integer_index(k if width is None else width)
        if not 1 <= k <= self.b or 3 * self.b > self.n:
            raise ValueError("require 1 <= k <= width and 3*width <= n")
        self.m = 3 * self.b
        search_bucket_size = integer_index(search_bucket_size)
        preconditioner_bucket_size = integer_index(preconditioner_bucket_size)
        if search_bucket_size < 0 or preconditioner_bucket_size < 0:
            raise ValueError("bucket sizes must be nonnegative")
        if search_bucket_size and not reuse_operator_images:
            raise ValueError("search buckets require reuse_operator_images=True")
        self.compact_projected_graphs = bool(compact_projected_graphs)
        self.reuse_operator_images = bool(reuse_operator_images)
        self._image_reuse_active = False
        self._search = None
        self.mass = mass if mass is not None else Identity(self.n, self.device)
        if preconditioner is not None and not hasattr(preconditioner, "apply"):
            if not all(
                hasattr(preconditioner, attr) for attr in ("shape", "device", "matvec", "dtype")
            ):
                raise ValueError("preconditioner must expose apply or upstream matvec metadata")
            preconditioner = UpstreamPreconditioner(preconditioner)
        self.preconditioner = (
            preconditioner if preconditioner is not None else Identity(self.n, self.device)
        )
        from .batched_cg import BatchedCGPreconditioner

        if isinstance(self.preconditioner, BatchedCGPreconditioner):
            if self.preconditioner.width != self.b:
                raise ValueError("batched preconditioner width must match the block width")
        for name, supplied in (("mass", self.mass), ("preconditioner", self.preconditioner)):
            if (
                not all(hasattr(supplied, attr) for attr in ("n", "device", "apply"))
                or supplied.n != self.n
                or supplied.device != self.device
            ):
                raise ValueError(f"{name} must match operator shape and device")
        self.initial_vectors, self.constraints = initial_vectors, constraints
        self.initial_perturbation = float(initial_perturbation)
        if not math.isfinite(self.initial_perturbation) or self.initial_perturbation < 0:
            raise ValueError("initial_perturbation must be finite and nonnegative")
        self.tol, self.rank_tol = float(tol), float(rank_tol)
        gram_chunk = integer_index(gram_chunk)
        if gram_chunk < 256 or gram_chunk % 256:
            raise ValueError("gram_chunk must be a positive multiple of 256")
        self.gram_chunk = gram_chunk
        self.symmetric_products = symmetric_products
        self.fused_updates = fused_updates
        self.paired_rotation = paired_rotation
        if not math.isfinite(tol) or tol <= 0 or not 0 < rank_tol < 1:
            raise ValueError("require finite tol > 0 and 0 < rank_tol < 1")
        if constraints is not None and (
            constraints.dtype != wp.float64
            or constraints.ndim != 2
            or constraints.shape[1] != self.n
            or constraints.device != self.device
            or constraints.shape[0] + self.b > self.n
        ):
            raise ValueError("constraints must be device FP64 rows with room for width modes")

        if initial_vectors is not None and (
            initial_vectors.dtype != wp.float64
            or initial_vectors.ndim != 2
            or initial_vectors.shape[1] != self.n
            or not 1 <= initial_vectors.shape[0] <= self.b
            or initial_vectors.device != self.device
        ):
            raise ValueError("initial_vectors must be device FP64 (1..width,n) rows")

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
        # Allocate before active-search branches shallow-copy this owner, so
        # every branch shares one setup-time workspace and replay allocates none.
        self._global_factor_workspace = (
            wp.empty((self.b, self.b), dtype=wp.float64, device=self.device)
            if self.device.is_cuda and 129 <= self.b <= 512
            else None
        )
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
        self.constraint_basis = None
        if constraints is not None:
            self.constraint_basis = MassOrthogonalComplementInverse(
                Identity(self.n, self.device),
                self.mass,
                constraints,
                normalize=normalize_constraints,
            )
            self.constraints = self.constraint_basis.basis
            self.bconstraints = self.constraint_basis.mass_basis
            self.projection = empty((self.m, constraints.shape[0]))
            self.projection_u = self.projection[self.b :]

        self.initial_norms = (
            empty(self.b) if initial_vectors is not None and self.initial_perturbation else None
        )

        if self.reuse_operator_images:
            self._image_x = self.az[: self.b]
            self._image_scratch = self.z[: self.b]
            self._image_constraints = (
                None if self.constraints is None else wp.empty_like(self.constraints)
            )
        if search_bucket_size:
            from ._block_search import ActiveSearch

            self._search = ActiveSearch(self, search_bucket_size, keep_history)
            self.search_widths, self.search_usage = self._search.widths, self._search.usage
        if preconditioner_bucket_size:
            from ._active_cg import DeviceBucketedCG

            if not isinstance(self.preconditioner, BatchedCGPreconditioner):
                raise TypeError("preconditioner buckets require BatchedCGPreconditioner")
            self.preconditioner = DeviceBucketedCG(
                self.preconditioner, self.errors, self.tol, preconditioner_bucket_size
            )

        self._configure_second_whitening()
        self._configure_first_whitening()
        self._configure_compact_projected_graphs()

    def _configure_compact_projected_graphs(self):
        if not self.compact_projected_graphs:
            return
        from ._compact_jacobi import CompactJacobi

        branches = [self] + ([] if self._search is None else self._search.buckets)
        for branch in branches:
            for projected in (branch.small, branch.metric):
                original = projected._global_jacobi
                # Small projected problems (including the Anvil recipe) retain
                # their established replay path. Odd/unsupported dimensions
                # retain the canonical solver selected during construction.
                if original is not None and 96 <= original.n <= 1024:
                    projected._global_jacobi = CompactJacobi(
                        original.h,
                        original.v,
                        original.cs,
                        original.ss,
                        original.partners,
                        original.temp,
                        original.vtemp,
                        rounds_per_loop=16,
                    )

    def _configure_second_whitening(self):
        # Only wide projected metrics on the qualified Ada path use this
        # specialization. Other dimensions/devices retain canonical whitening.
        branches = [self] + ([] if self._search is None else self._search.buckets)
        minimum = getattr(self, "_metric_whitening_minimum", 96)
        architectures = getattr(self, "_metric_whitening_arches", (89,))
        eligible = []
        for branch in branches:
            branch._second_whitener = None
            if (
                self.device.is_cuda
                and (architectures is None or self.device.arch in architectures)
                and minimum <= branch.su.shape[0] <= 512
            ):
                eligible.append(branch)
        self._second_whitening_storage = None
        self._second_whitening_usage = None
        if eligible:
            from ._block_whitening import SecondWhitening

            size = max(branch.su.shape[0] for branch in eligible)
            self._second_whitening_storage = tuple(
                wp.empty((size, size), dtype=wp.float64, device=self.device) for _ in range(3)
            )
            self._second_whitening_usage = wp.zeros(2, dtype=wp.int32, device=self.device)
            for branch in eligible:
                branch._second_whitener = SecondWhitening(
                    branch.su.shape[0],
                    self.device,
                    self._second_whitening_usage,
                    self._second_whitening_storage,
                )

    def _configure_first_whitening(self):
        branches = [self] + ([] if self._search is None else self._search.buckets)
        minimum = getattr(self, "_first_whitening_minimum", 96)
        architectures = getattr(self, "_first_whitening_arches", (89,))
        enabled = getattr(self, "_first_whitening_enabled", True)
        eligible = []
        for branch in branches:
            branch._first_whitener = None
            if (
                enabled
                and self.device.is_cuda
                and (architectures is None or self.device.arch in architectures)
                and minimum <= branch.su.shape[0] <= 512
            ):
                eligible.append(branch)
        self._first_whitening_storage = None
        self._first_whitening_usage = None
        if eligible:
            from ._first_whitening import FirstWhitening

            size = max(branch.su.shape[0] for branch in eligible)
            storage = self._second_whitening_storage
            if storage is None or storage[0].shape[0] < size:
                storage = tuple(
                    wp.empty((size, size), dtype=wp.float64, device=self.device) for _ in range(3)
                )
            self._first_whitening_storage = storage
            self._first_whitening_usage = wp.zeros(3, dtype=wp.int32, device=self.device)
            for branch in eligible:
                branch._first_whitener = FirstWhitening(
                    branch, storage, self._first_whitening_usage
                )

    def launch(self, kernel, shape, args):
        if (
            kernel is K.triangular
            and getattr(self, "_register_triangular_enabled", True)
            and self.device.is_cuda
            and self.device.arch == 89
            and 96 <= args[0].shape[0] <= 256
        ):
            # Interleave serial row recurrences to reuse earlier solved values.
            # Keep both QR passes; no inverse or additional workspace is needed.
            # Other widths/devices retain the established scalar application.
            from ._triangular import register_triangular

            wp.launch(register_triangular, shape, args, block_dim=256, device=self.device)
            return
        if kernel is K.cholesky:
            size = args[0].shape[0]
            # Tiny factors are not a meaningful whole-solve cost (including
            # Anvil's width 24). Keep their existing path, and only specialize
            # qualified wide blocks with shared or preallocated global storage.
            required_shared = 8 * (size * (size + 1) // 2 + 1)
            if (
                self.device.is_cuda
                and 48 <= size <= 128
                and required_shared + 1024 <= self.device.max_shared_memory_per_block
            ):
                from ._projected_cholesky import packed_cholesky_kernel

                wp.launch_tiled(
                    packed_cholesky_kernel(size),
                    dim=1,
                    inputs=args,
                    block_dim=128,
                    device=self.device,
                )
                return
            work = getattr(self, "_global_factor_workspace", None)
            if self.device.is_cuda and 129 <= size <= 512 and work is not None:
                from ._projected_cholesky import global_cholesky

                wp.launch_tiled(
                    global_cholesky,
                    dim=1,
                    inputs=[*args, work],
                    block_dim=256 if size <= 256 else 512,
                    device=self.device,
                )
                return
        wp.launch(kernel, shape, args, device=self.device)

    def gram(self, x, y, out, symmetric=False):
        gram(x, y, out, chunk=self.gram_chunk, symmetric=symmetric and self.symmetric_products)

    def subtract(self, basis, coeff, target, scratch):
        if self.fused_updates:
            from .block_products import subtract_projection

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
                if self._image_reuse_active:
                    self.subtract(
                        self._image_constraints,
                        self.projection[: self.b],
                        self._image_x,
                        self._image_scratch,
                    )

    def orthogonalize_x(self):
        for i in range(2):
            self.mass.apply(self.sx, self.bx)
            self.gram(self.sx, self.bx, self.gx, symmetric=True)
            if i == 0:
                self.launch(scales, self.b, [self.gx, self.xscale, self.status])
                self.launch(scale_rows, self.sx.shape, [self.sx, self.xscale])
                if self._image_reuse_active:
                    self.launch(scale_rows, self.sx.shape, [self._image_x, self.xscale])
                self.launch(normalize_gram, self.gx.shape, [self.gx, self.xscale])
            self.launch(K.cholesky, 1, [self.gx, self.lx, self.status])
            self.launch(K.triangular, self.n, [self.sx, self.lx])
            if self._image_reuse_active:
                self.launch(K.triangular, self.n, [self._image_x, self.lx])
        self.mass.apply(self.sx, self.bx)

    def orthogonalize_search(self, x):
        for _ in range(2):
            self.gram(x, self.bx, self.projection_x)
            self.subtract(self.sx, self.projection_x, x, self.bu)
            if self.constraints is not None:
                self.gram(x, self.bconstraints, self.projection_u)
                self.subtract(self.constraints, self.projection_u, x, self.bu)

    def whiten(self, x, out, normalize=False):
        if (
            normalize
            and self._first_whitener is not None
            and wp.get_stream(self.device).is_capturing
        ):
            self._first_whitener.apply(self, x, out)
            return
        if (
            not normalize
            and self._second_whitener is not None
            and wp.get_stream(self.device).is_capturing
        ):
            self._second_whitener.apply(self, x, out)
            return
        self.mass.apply(x, self.bu)
        self.gram(x, self.bu, self.metric.g, symmetric=True)
        if normalize:
            self.launch(scales, x.shape[0], [self.metric.g, self.scale, self.status])
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
        if self._image_reuse_active:
            self.op.apply(self.s[self.b :], self.az[self.b :])
        else:
            self.op.apply(self.s, self.az)
        self.gram(self.s, self.az, self.small.g, symmetric=True)
        self.launch(mask_projected, 1, [self.small.g, self.active, self.rank, self.status, self.b])
        self.small.which = 0
        self.small.diagonalize()
        self.launch(
            K.rotation_coefficients, self.coeff.shape, [self.small.v, self.small.order, self.coeff]
        )
        if self.paired_rotation:
            from .block_products import paired_ritz_rotation

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
        if self._first_whitening_usage is not None:
            self._first_whitening_usage.zero_()
        if self._second_whitening_usage is not None:
            self._second_whitening_usage.zero_()
        self._image_reuse_active = False
        self.status.zero_()
        self.iterations.zero_()
        if hasattr(self.preconditioner, "reset_history"):
            self.preconditioner.reset_history()
        if hasattr(self.preconditioner, "reset_stats"):
            self.preconditioner.reset_stats()
        self.s.zero_()
        if self.constraint_basis is not None:
            self.constraint_basis.refresh()
            wp.copy(self.status, self.constraint_basis.basis_status)
        if (
            self.initial_vectors is None
            or self.initial_vectors.shape[0] < self.b
            or self.initial_perturbation
        ):
            self.launch(K.random_block, self.sx.shape, [self.sx, seed])
        if self.initial_vectors is not None:
            if self.initial_perturbation:
                from .block_products import initial_row_norms, perturb_initial_rows

                self.initial_norms.zero_()
                wp.launch_tiled(
                    initial_row_norms,
                    dim=(self.initial_vectors.shape[0], (self.n + 255) // 256),
                    inputs=[self.initial_vectors, self.initial_norms],
                    block_dim=128,
                    device=self.device,
                )
                self.launch(
                    perturb_initial_rows,
                    self.initial_vectors.shape,
                    [self.initial_vectors, self.initial_norms, self.sx, self.initial_perturbation],
                )
            else:
                wp.copy(self.sx[: self.initial_vectors.shape[0]], self.initial_vectors)
        self.launch(K.identity_order, self.b, [self.order])
        self.rayleigh_ritz()
        if self.reuse_operator_images and self.constraints is not None:
            self.op.apply(self.constraints, self._image_constraints)
        if self._search is not None:
            self._search.usage.zero_()

    def step(self):
        if self._search is not None:
            self._search.step()
            return
        if self.reuse_operator_images:
            wp.copy(self._image_x, self.ax)
            self._image_reuse_active = True
        try:
            wp.copy(self.sx, self.x)
            self.preconditioner.apply(self.r, self.sw)
            wp.copy(self.sp, self.p)
            self.rayleigh_ritz()
        finally:
            self._image_reuse_active = False

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
        """Run an explicit fixed budget; use capture() for adaptive device stopping."""
        steps = integer_index(steps)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        self.initialize()
        for _ in range(steps):
            self.step()
        self.iterations.fill_(steps)
        return self

    def capture(self, max_iterations=100, adaptive=None, finalizer=None, *, initializer=None):
        """Capture optional input preparation, the solve, and postprocessing.

        Callbacks record preallocated device operations during setup and are
        warmed eagerly before capture. They do not execute as Python callbacks
        during replay. The initializer runs before initialization (including
        mass orthonormalization); the finalizer runs after the solve.
        """
        max_iterations = integer_index(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        static_only = any(
            getattr(op, "requires_static_outer_capture", False)
            for op in (self.op, self.mass, self.preconditioner)
        )
        if adaptive is None:
            adaptive = not static_only
        if adaptive and static_only:
            raise ValueError("This preconditioner requires adaptive=False for the outer graph")
        if initializer is not None:
            initializer()
        self.solve(1)
        if self._search is not None:
            self._search.warm()
        if finalizer is not None:
            finalizer()
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
            if initializer is not None:
                initializer()
            self.initialize()
            self.running.fill_(1)
            if adaptive:
                wp.capture_while(self.running, body)
            else:
                for _ in range(max_iterations):
                    self.step()
                self.iterations.fill_(max_iterations)
            if finalizer is not None:
                finalizer()
        self.graph = capture.graph
        return self.graph
