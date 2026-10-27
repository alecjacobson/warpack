"""Device evidence for a supplied Ritz space; never an exact nullity certificate."""

import math

import warp as wp

from . import kernels as K
from .generalized import generalized_residual
from .generalized_krylov import quotient_parts, quotients


@wp.kernel
def _classify(
    values: wp.array[wp.float64],
    sums: wp.array2d[wp.float64],
    norms: wp.array2d[wp.float64],
    residuals: wp.array[wp.float64],
    bounds: wp.array[wp.float64],
    classes: wp.array[wp.int32],
    zero_threshold: wp.float64,
    positive_threshold: wp.float64,
    mass_lower_bound: wp.array[wp.float64],
    residual_tolerance: wp.float64,
    identity_mass: bool,
):
    j = wp.tid()
    value = values[j]
    metric_norm = sums[j, 1]
    residual = wp.sqrt(norms[j, 0])
    denominator = wp.max(
        wp.float64(1.0), wp.sqrt(norms[j, 1]) + wp.abs(value) * wp.sqrt(norms[j, 2])
    )
    relative = residual / denominator
    lower = mass_lower_bound[0]
    error = residual / wp.sqrt(lower * metric_norm)
    residuals[j] = relative
    bounds[j] = error
    state = int(2)  # Ambiguous between zero and a positive guard.
    if (
        not wp.isfinite(lower)
        or lower <= wp.float64(0.0)
        or (identity_mass and lower > wp.float64(1.0))
        or not wp.isfinite(value)
        or not wp.isfinite(relative)
        or not wp.isfinite(error)
        or metric_norm <= wp.float64(0.0)
    ):
        state = 4  # Invalid data.
    elif relative > residual_tolerance:
        state = 3  # Physical residual has not converged.
    elif value + error < -zero_threshold:
        state = 5  # Contradicts the assumed PSD spectrum beyond the zero band.
    elif wp.abs(value) + error <= zero_threshold:
        state = 0  # Numerically near zero, not a topological-zero assertion.
        if value - error > wp.float64(0.0):
            state = 6  # Positive residual interval entirely inside the zero band.
    elif value - error >= positive_threshold:
        state = 1
    classes[j] = state


@wp.kernel
def _summarize(
    gram: wp.array2d[wp.float64],
    values: wp.array[wp.float64],
    bounds: wp.array[wp.float64],
    classes: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    guard: wp.array[wp.float64],
    flags: wp.array[wp.int32],
    suggested: wp.array[wp.int32],
    orthogonality_error: wp.array[wp.float64],
    tolerance: wp.float64,
    expected_nullity: int,
    dimension: int,
):
    capacity = classes.shape[0]
    state = int(1)  # Completeness is always unverified by floating Ritz data alone.
    zero_count, ambiguous, weak_positive = int(0), int(0), int(0)
    positive = wp.float64(wp.inf)
    orth_error = wp.float64(0.0)
    for i in range(capacity):
        for j in range(capacity):
            expected = wp.float64(0.0)
            if i == j:
                expected = wp.float64(1.0)
            if not wp.isfinite(gram[i, j]):
                state = state | 2 | 4
                orth_error = wp.float64(wp.inf)
            else:
                orth_error = wp.max(orth_error, wp.abs(gram[i, j] - expected))
        kind = classes[i]
        if kind == 0 or kind == 6:
            zero_count += 1
            if kind == 6:
                weak_positive += 1
        elif kind == 1:
            positive = wp.min(positive, values[i] - bounds[i])
        else:
            ambiguous += 1
            if kind == 3:
                state = state | 8
            elif kind == 4:
                state = state | 4
            elif kind == 5:
                state = state | 16
    if orth_error > tolerance:
        state = state | 2
    if (state & 6) != 0:
        # Duplicate/invalid vectors cannot establish independent numerical modes.
        zero_count, weak_positive, ambiguous = 0, 0, capacity
        positive = wp.float64(wp.nan)
    if ambiguous > 0 or weak_positive > 0 or not wp.isfinite(positive):
        state = state | 64
    next_capacity = capacity
    if (state & 6) == 0 and (zero_count == capacity or expected_nullity >= capacity):
        state = state | 32
        next_capacity = wp.min(dimension, wp.max(2 * capacity + 1, expected_nullity + 1))
    if expected_nullity >= 0 and zero_count != expected_nullity:
        state = state | 128
    if capacity == dimension:
        state = state | 256
    counts[0] = zero_count
    counts[1] = ambiguous
    counts[2] = weak_positive
    guard[0] = positive
    flags[0] = state
    suggested[0] = next_capacity
    orthogonality_error[0] = orth_error


class NumericalNullspaceDiagnostics:
    """Evaluate a fixed-capacity Ritz space of a real symmetric PSD pencil.

    This reports evidence about the supplied vectors, **not total nullity**.
    ``COMPLETENESS_UNVERIFIED`` is always set: small residuals, independent
    vectors and a positive guard do not exclude an omitted repeated direction.
    Even an optional externally known ``expected_nullity`` only checks agreement;
    it is not a proof derived from the floating eigensolve.

    Rows of ``vectors`` must be M-orthonormal FP64 arrays on the operator device.
    The entrywise Gram tolerance must be less than 1/capacity to preserve a
    positive independence margin.
    ``mass=None`` means identity. For generalized problems, supply a positive
    lower spectral bound on SPD M, as a scalar or one FP64 device value.
    A supplied device bound is never read on the host; changing it in place is
    reflected on replay, including invalid-bound diagnostics. Its mathematical
    validity for a general mass remains the caller's responsibility.
    Residual-based eigenvalue error estimates are
    ||Ax-lambda Mx|| / sqrt(mass_lower_bound * x.T M x). In exact arithmetic
    these bound distance to some eigenvalue, not multiplicity or its index.
    Floating estimates are not rigorous interval arithmetic certificates.

    ``numerical_count`` counts returned intervals wholly within the zero band.
    ``weak_positive_count`` separately identifies those wholly above zero;
    those are not silently asserted to be topological zeros. ``ambiguous_count``
    counts other unclassified/unconverged/invalid rows. ``positive_guard`` is
    the smallest lower endpoint above ``positive_threshold`` (inf if absent,
    NaN for an invalid metric basis). Thresholds have eigenvalue units; choose
    them for the matrix scaling. ``residual_tolerance`` uses the same normalized
    physical residual as GeneralizedKrylovSchur.

    ``flags`` is a device bitmask. Invalid metric independence suppresses counts.
    ``suggested_capacity`` is a device hint when all supplied rows are near zero
    or known nullity fills the space; it never reallocates or runs a solver.
    GAP_UNRESOLVED also covers a missing guard or a resolved weak positive mode
    inside the zero band. At full matrix dimension, AT_MAX_CAPACITY is set.

    Construction preallocates workspace; ``run()`` is graph capturable and has
    no numerical host readback. Changed vector/operator values are recomputed
    on replay. Replacing buffers or changing scalar configuration needs a new
    diagnostic/capture. Do not share its scratch across concurrent streams.

    Classification codes: 0 near zero, 1 positive guard, 2 ambiguous,
    3 unconverged, 4 invalid data, 5 negative spectrum, 6 weak positive inside
    the zero band. Flag constants are bit masks and may be combined.

    Args:
        operator: Real symmetric PSD operator.
        vectors: Borrowed FP64 device rows (capacity, n), mass-orthonormal.
        mass: Optional SPD real metric operator; None means identity.
        mass_lower_bound: Positive lower spectral bound on M, as a scalar or FP64 device array
            (1,). Required with a nonidentity mass; identity defaults to one.
        zero_threshold: Nonnegative near-zero spectral band in eigenvalue units.
        positive_threshold: Positive guard threshold greater than zero_threshold, in eigenvalue
            units.
        residual_tolerance: Positive normalized original-pencil residual tolerance.
        orthogonality_tolerance: Positive entrywise mass-Gram tolerance, less than 1/capacity.
        expected_nullity: Optional externally known nonnegative nullity for consistency
            checking; not a computed certificate.

    Attributes:
        values: FP64 device array (capacity,), Rayleigh values.
        residuals: FP64 device array (capacity,), physical residuals.
        error_bounds: FP64 device array (capacity,), residual-based eigenvalue distance
            estimates.
        classification: int32 device array (capacity,), per-row classification.
        numerical_count: int32 device array (1,), returned intervals wholly inside the zero
            band.
        ambiguous_count: int32 device array (1,), unclassified/unconverged/invalid rows.
        weak_positive_count: int32 device array (1,), positive intervals inside the zero band.
        positive_guard: FP64 device array (1,), smallest lower endpoint above the positive
            threshold.
        flags: int32 device array (1,), diagnostic bitmask; COMPLETENESS_UNVERIFIED is always
            set.
        suggested_capacity: int32 device array (1,), capacity hint only; never reallocates or
            solves.
        orthogonality_error: FP64 device array (1,), entrywise mass-Gram error.
    """

    COMPLETENESS_UNVERIFIED = 1
    INVALID_BASIS = 2
    INVALID_DATA = 4
    UNCONVERGED = 8
    NEGATIVE_SPECTRUM = 16
    CAPACITY_EXHAUSTED = 32
    GAP_UNRESOLVED = 64
    EXPECTED_COUNT_MISMATCH = 128
    AT_MAX_CAPACITY = 256

    def __init__(
        self,
        operator,
        vectors,
        mass=None,
        *,
        mass_lower_bound=None,
        zero_threshold=1e-8,
        positive_threshold=1e-7,
        residual_tolerance=1e-8,
        orthogonality_tolerance=1e-8,
        expected_nullity=None,
    ):
        self.operator, self.mass, self.vectors = operator, mass, vectors
        self.n, self.device = operator.n, operator.device
        if vectors.ndim != 2 or vectors.shape[1] != self.n or not 1 <= vectors.shape[0] <= self.n:
            raise ValueError("vectors must have shape (capacity, n), with 1 <= capacity <= n")
        if vectors.dtype != wp.float64 or vectors.device != self.device:
            raise ValueError("vectors must be FP64 on the operator device")
        if mass is not None and (mass.n != self.n or mass.device != self.device):
            raise ValueError("mass must match the operator dimension and device")
        if mass_lower_bound is None:
            if mass is not None:
                raise ValueError("generalized diagnostics require a lower spectral bound on M")
            mass_lower_bound = 1.0
        for value in (
            zero_threshold,
            positive_threshold,
            residual_tolerance,
            orthogonality_tolerance,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("bounds and tolerances must be finite and positive")
        if isinstance(mass_lower_bound, wp.array):
            if (
                mass_lower_bound.ndim != 1
                or mass_lower_bound.shape != (1,)
                or mass_lower_bound.dtype != wp.float64
                or mass_lower_bound.device != self.device
            ):
                raise ValueError("device mass bound must be one FP64 value on the operator device")
            self.mass_lower_bound = mass_lower_bound
        else:
            if not math.isfinite(mass_lower_bound) or mass_lower_bound <= 0:
                raise ValueError("mass_lower_bound must be finite and positive")
            if mass is None and mass_lower_bound > 1:
                raise ValueError("a lower spectral bound for identity mass cannot exceed one")
            self.mass_lower_bound = wp.array(
                [float(mass_lower_bound)], dtype=wp.float64, device=self.device
            )
        if positive_threshold <= zero_threshold:
            raise ValueError("positive_threshold must exceed zero_threshold")
        if expected_nullity is not None and (
            not isinstance(expected_nullity, int) or not 0 <= expected_nullity <= self.n
        ):
            raise ValueError("expected_nullity must be an integer between zero and n")
        self.capacity = vectors.shape[0]
        if orthogonality_tolerance >= 1.0 / self.capacity:
            raise ValueError("orthogonality_tolerance must be smaller than 1/capacity")
        self.zero_threshold, self.positive_threshold = (
            float(zero_threshold),
            float(positive_threshold),
        )
        self.residual_tolerance = float(residual_tolerance)
        self.orthogonality_tolerance = float(orthogonality_tolerance)
        self.expected_nullity = -1 if expected_nullity is None else expected_nullity
        kw = dict(dtype=wp.float64, device=self.device)
        self.ax, self.mx = wp.empty_like(vectors), wp.empty_like(vectors)
        self.sums = wp.zeros((self.capacity, 2), **kw)
        self.norms = wp.zeros((self.capacity, 3), **kw)
        self.gram = wp.zeros((self.capacity, self.capacity), **kw)
        self.values = wp.empty(self.capacity, **kw)
        self.residuals, self.error_bounds = wp.empty_like(self.values), wp.empty_like(self.values)
        self.classification = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.order = wp.empty_like(self.classification)
        wp.launch(K.identity_order, self.capacity, [self.order], device=self.device)
        self.counts = wp.zeros(3, dtype=wp.int32, device=self.device)
        self.numerical_count = self.counts[0:1]
        self.ambiguous_count = self.counts[1:2]
        self.weak_positive_count = self.counts[2:3]
        self.flags = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.suggested_capacity = wp.empty_like(self.flags)
        self.positive_guard, self.orthogonality_error = wp.empty(1, **kw), wp.empty(1, **kw)
        self.views = [
            (vectors[j : j + 1], self.ax[j : j + 1], self.mx[j : j + 1])
            for j in range(self.capacity)
        ]

    def run(self):
        """Evaluate current inputs and update device diagnostics; graph capturable.

        Returns:
            None. Results are written to the documented device arrays.
        """
        for x, ax, mx in self.views:
            self.operator.apply(x, ax)
            if self.mass is None:
                wp.copy(mx, x)
            else:
                self.mass.apply(x, mx)
        self.sums.zero_()
        self.norms.zero_()
        self.gram.zero_()
        chunks = (self.n + 255) // 256
        wp.launch_tiled(
            quotient_parts,
            dim=(self.capacity, chunks),
            inputs=[self.vectors, self.ax, self.mx, self.sums],
            block_dim=128,
            device=self.device,
        )
        wp.launch(quotients, self.capacity, [self.sums, self.values], device=self.device)
        wp.launch_tiled(
            generalized_residual,
            dim=(self.capacity, chunks),
            inputs=[self.vectors, self.ax, self.mx, self.values, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch_tiled(
            K.gram_partial,
            dim=(self.capacity, self.capacity, chunks),
            inputs=[self.vectors, self.mx, self.gram],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            _classify,
            self.capacity,
            [
                self.values,
                self.sums,
                self.norms,
                self.residuals,
                self.error_bounds,
                self.classification,
                self.zero_threshold,
                self.positive_threshold,
                self.mass_lower_bound,
                self.residual_tolerance,
                self.mass is None,
            ],
            device=self.device,
        )
        wp.launch(
            _summarize,
            1,
            [
                self.gram,
                self.values,
                self.error_bounds,
                self.classification,
                self.counts,
                self.positive_guard,
                self.flags,
                self.suggested_capacity,
                self.orthogonality_error,
                self.orthogonality_tolerance,
                self.expected_nullity,
                self.n,
            ],
            device=self.device,
        )
