"""Optional numerical completeness checks for symmetric SPD-mass pencils."""

import math

import warp as wp
import warp.sparse as ws

from . import kernels as K
from .cudss_diagnostics import InertiaResult
from .generalized import generalized_residual
from .solver import CuDSSInverse, SparseOperator


@wp.kernel
def _check(
    values: wp.array[wp.float64],
    norms: wp.array2d[wp.float64],
    gram: wp.array2d[wp.float64],
    projected: wp.array2d[wp.float64],
    count: wp.array[wp.int32],
    residuals: wp.array[wp.float64],
    flags: wp.array[wp.int32],
    certified: wp.array[wp.int32],
    orth_error: wp.array[wp.float64],
    gap: wp.array[wp.float64],
    upper: wp.float64,
    residual_tol: wp.float64,
    orth_tol: wp.float64,
    gap_tol: wp.float64,
):
    state = int(0)
    if count[0] == 1:
        state = state | 1
    elif count[0] != 0:
        state = state | 2
    elif count[1] != values.shape[0]:
        state = state | 4
    error = wp.float64(0.0)
    max_upper = wp.float64(-wp.inf)
    scale = wp.max(wp.float64(1.0), wp.abs(upper))
    for i in range(values.shape[0]):
        value = values[i]
        relative = wp.sqrt(norms[i, 0]) / wp.max(
            wp.float64(1.0), wp.sqrt(norms[i, 1]) + wp.abs(value) * wp.sqrt(norms[i, 2])
        )
        residuals[i] = relative
        if not wp.isfinite(value) or not wp.isfinite(relative):
            state = state | 64
        elif relative > residual_tol:
            state = state | 16
        scale = wp.max(scale, wp.abs(value))
        # Gershgorin sufficient test for negative definiteness of the whole
        # restricted form V (K-upper*M) V.T, not just its diagonal entries.
        row_upper = projected[i, i] - upper * gram[i, i]
        for j in range(values.shape[0]):
            target = wp.float64(0.0)
            if i == j:
                target = wp.float64(1.0)
            if not wp.isfinite(gram[i, j]) or not wp.isfinite(projected[i, j]):
                state = state | 64
            error = wp.max(error, wp.abs(gram[i, j] - target))
            if i != j:
                row_upper += wp.abs(projected[i, j] - upper * gram[i, j])
        max_upper = wp.max(max_upper, row_upper)
    if error > orth_tol:
        state = state | 8
    if not wp.isfinite(max_upper) or max_upper >= -gap_tol * scale:
        state = state | 32
    orth_error[0] = error
    gap[0] = -max_upper
    flags[0] = state
    certified[0] = int(state == 0)


class LowestModeCertificate:
    """Numerical coverage of *all* eigenpairs below a supplied spectral gap.

    K must be real symmetric and M SPD (identity if omitted). Matrices are
    compressed scalar FP64 Warp CSR. ``vectors`` are (k,n) physical rows;
    ``values`` are their original-pencil eigenvalues. Optional ``nullspace``
    supplies q known zero modes as M-orthonormal rows. We check all k+q rows
    together, including their residuals and independence; no trusted integer
    nullity substitutes for this basis.

    ``upper_bound`` must lie in a gap above the requested set. Construction
    factors K-upper_bound*M with cuDSS and queries numerical inertia on the
    host. This explicit setup step is NOT capturable. ``run()`` is entirely
    Warp and graph capturable: fresh original residuals, M-Gram independence,
    a conservative whole-subspace gap test and agreement with the count.
    No automatic gap search, eigensolve, or retry is performed.

    ``certified[0]==1`` means numerical checks pass under the symmetry/SPD
    preconditions, not an interval-arithmetic proof. cuDSS errors, nonzero
    pivot counters and unavailable counts can never certify. ``backend=None``
    performs only the Warp checks and always flags COUNT_UNAVAILABLE.

    Flags: COUNT_UNAVAILABLE=1, COUNT_INCONCLUSIVE=2, COUNT_MISMATCH=4,
    INVALID_BASIS=8, UNCONVERGED=16, GAP_UNRESOLVED=32, INVALID_DATA=64.
    Residuals use ||Kv-lambda Mv|| / max(1, ||Kv||+|lambda| ||Mv||).
    Gap tolerance is relative to max(1, |upper_bound|, max |values|).
    Like other numerical tolerances, these depend on matrix units/scaling.

    After changing K or M values, call ``refresh_count()`` outside capture
    before replay; it invalidates the old result and replaces metadata in
    existing device storage. Changing vectors/values/nullspace values needs
    only run(). Topology/buffer identities and upper_bound are fixed. Do not
    mutate inputs concurrently or use one instance on concurrent streams.

    Args:
        stiffness: Compressed scalar FP64 symmetric Warp BsrMatrix K.
        vectors: Borrowed FP64 physical rows (k, n), jointly mass-orthonormal with nullspace.
        values: Borrowed FP64 device array (k,) of original-pencil eigenvalues.
        upper_bound: Fixed real gap location above every requested mode; not found
            automatically.
        mass: Optional compressed scalar FP64 SPD Warp BsrMatrix M; None means identity.
        nullspace: Optional FP64 device rows (q, n) of known mass-orthonormal zero modes,
            excluded from vectors.
        backend: cudss for a setup inertia query, or None for residual/independence checks
            without certification.
        residual_tolerance: Positive tolerance for normalized original-pencil residuals.
        orthogonality_tolerance: Positive entrywise tolerance for the combined basis mass-Gram
            matrix.
        gap_tolerance: Positive relative separation tolerance at upper_bound.

    Attributes:
        certified: int32 device array (1,), one only when all numerical coverage checks pass.
        flags: int32 device array (1,), bitmask described above.
        residuals: FP64 device array (k+q,), original-pencil residuals for returned and known
            null modes.
        orthogonality_error: FP64 device array (1,), combined mass-Gram error.
        gap_margin: FP64 device array (1,), tested subspace gap margin.
        inertia: Host InertiaResult from the latest setup count query.
    """

    COUNT_UNAVAILABLE = 1
    COUNT_INCONCLUSIVE = 2
    COUNT_MISMATCH = 4
    INVALID_BASIS = 8
    UNCONVERGED = 16
    GAP_UNRESOLVED = 32
    INVALID_DATA = 64

    def __init__(
        self,
        stiffness,
        vectors,
        values,
        *,
        upper_bound,
        mass=None,
        nullspace=None,
        backend="cudss",
        residual_tolerance=1e-8,
        orthogonality_tolerance=1e-8,
        gap_tolerance=1e-10,
    ):
        self.stiffness, self.mass_matrix = stiffness, mass
        self.n, self.device = stiffness.shape[0], stiffness.device
        if not self.device.is_cuda:
            raise ValueError("LowestModeCertificate requires CUDA matrices")
        if wp.get_stream(self.device).is_capturing:
            raise RuntimeError("certificate construction is setup and cannot run during capture")
        for a in (stiffness, mass):
            if a is not None and (
                a.shape != (self.n, self.n)
                or a.device != self.device
                or a.values.dtype != wp.float64
                or a.row_counts is not None
            ):
                raise ValueError("matrices must be matching compressed scalar FP64 CSR")
        if backend not in (None, "cudss"):
            raise ValueError("backend must be 'cudss' or None")
        if not math.isfinite(upper_bound):
            raise ValueError("upper_bound must be finite")
        for x in (residual_tolerance, orthogonality_tolerance, gap_tolerance):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("tolerances must be finite and positive")
        for rows in (vectors, nullspace):
            if rows is not None and (
                rows.ndim != 2
                or rows.shape[1] != self.n
                or rows.dtype != wp.float64
                or rows.device != self.device
            ):
                raise ValueError("basis rows must be FP64 (k,n) on the matrix device")
        self.k = vectors.shape[0]
        self.q = 0 if nullspace is None else nullspace.shape[0]
        self.capacity = self.k + self.q
        if self.k < 1 or self.capacity > self.n:
            raise ValueError("require 1 <= k <= k+nullity <= n")
        if values.shape != (self.k,) or values.dtype != wp.float64 or values.device != self.device:
            raise ValueError("values must be matching device FP64 eigenvalues")
        if orthogonality_tolerance >= 1 / self.capacity:
            raise ValueError("orthogonality_tolerance must be less than 1/capacity")
        self.vectors, self.values, self.nullspace = vectors, values, nullspace
        self.backend, self.upper_bound = backend, float(upper_bound)
        self.residual_tolerance = float(residual_tolerance)
        self.orthogonality_tolerance = float(orthogonality_tolerance)
        self.gap_tolerance = float(gap_tolerance)
        self.op = SparseOperator(stiffness)
        self.mass = None if mass is None else SparseOperator(mass)
        kw = dict(dtype=wp.float64, device=self.device)
        self.rows = wp.empty((self.capacity, self.n), **kw)
        self.all_values = wp.zeros(self.capacity, **kw)
        self.ax, self.mx = wp.empty_like(self.rows), wp.empty_like(self.rows)
        self.gram = wp.zeros((self.capacity, self.capacity), **kw)
        self.projected = wp.zeros_like(self.gram)
        self.norms = wp.zeros((self.capacity, 3), **kw)
        self.residuals = wp.empty(self.capacity, **kw)
        self.orthogonality_error, self.gap_margin = wp.empty(1, **kw), wp.empty(1, **kw)
        self.order = wp.empty(self.capacity, dtype=wp.int32, device=self.device)
        wp.launch(K.identity_order, self.capacity, [self.order], device=self.device)
        self.flags = wp.ones(1, dtype=wp.int32, device=self.device)
        self.certified = wp.zeros_like(self.flags)
        self._count = wp.array([1, -1], dtype=wp.int32, device=self.device)
        self.refresh_count()

    def refresh_count(self):
        """Refactor the current pencil and replace count metadata outside capture.

        Returns:
            The new InertiaResult host snapshot. Device count metadata is updated
            in place; this synchronizing setup method is not graph capturable.
        """
        if wp.get_stream(self.device).is_capturing:
            raise RuntimeError("refresh_count is setup work and cannot run during capture")
        self.certified.zero_()
        self.flags.fill_(self.COUNT_UNAVAILABLE)
        self._count.assign([1, -1])
        self.inertia = InertiaResult("unavailable", reason="no inertia backend requested")
        if self.backend is not None:
            shifted = ws.bsr_copy(self.stiffness)
            mass = self.mass_matrix
            if mass is None:
                mass = ws.bsr_diag(wp.float64(1.0), self.n, device=self.device)
            ws.bsr_axpy(mass, shifted, alpha=-self.upper_bound)
            inverse = CuDSSInverse(shifted, 1, mtype="symmetric", refinement_steps=0)
            try:
                self.inertia = inverse.inertia()
            finally:
                inverse.close()
        state = {"available": 0, "unavailable": 1, "inconclusive": 2}[self.inertia.status]
        negative = -1 if self.inertia.negative is None else self.inertia.negative
        self._count.assign([state, negative])
        return self.inertia

    def run(self):
        """Evaluate current inputs and update device diagnostics; graph capturable.

        Returns:
            The certified device array (int32, shape (1,)); no implicit host readback.
        """
        if self.q:
            wp.copy(self.rows, self.nullspace, count=self.q * self.n)
        wp.copy(self.rows, self.vectors, dest_offset=self.q * self.n, count=self.k * self.n)
        wp.copy(self.all_values, self.values, dest_offset=self.q, count=self.k)
        self.op.apply(self.rows, self.ax)
        if self.mass is None:
            wp.copy(self.mx, self.rows)
        else:
            self.mass.apply(self.rows, self.mx)
        self.norms.zero_()
        self.gram.zero_()
        self.projected.zero_()
        chunks = (self.n + 255) // 256
        wp.launch_tiled(
            generalized_residual,
            dim=(self.capacity, chunks),
            inputs=[self.rows, self.ax, self.mx, self.all_values, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        for image, out in ((self.mx, self.gram), (self.ax, self.projected)):
            wp.launch_tiled(
                K.gram_partial,
                dim=(self.capacity, self.capacity, chunks),
                inputs=[self.rows, image, out],
                block_dim=128,
                device=self.device,
            )
        wp.launch(
            _check,
            1,
            [
                self.all_values,
                self.norms,
                self.gram,
                self.projected,
                self._count,
                self.residuals,
                self.flags,
                self.certified,
                self.orthogonality_error,
                self.gap_margin,
                self.upper_bound,
                self.residual_tolerance,
                self.orthogonality_tolerance,
                self.gap_tolerance,
            ],
            device=self.device,
        )
        return self.certified
