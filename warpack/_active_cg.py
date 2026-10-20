"""Device-selected CG batch buckets for fresh LOBPCG residuals.

Shape selection is encoded in CUDA conditional nodes, never a
residual readback. A direction can reactivate after any Ritz update. All CG
arithmetic is upstream Warp; this adapter only packs rows and selects a
preallocated state. Outside capture the same packed solve uses the full-width
state, avoiding capture_if's host condition readback during ordinary warmup.
"""

import math
from operator import index

import warp as wp

from .batched_cg import merge_statistics


@wp.kernel
def select_rows(
    errors: wp.array[wp.float64],
    tolerance: wp.float64,
    rows: wp.array[wp.int32],
    count: wp.array[wp.int32],
    widths: wp.array[wp.int32],
    conditions: wp.array[wp.int32],
):
    # Width is small. Deterministic order avoids an atomic permutation of RHSs.
    active = int(0)
    for i in range(errors.shape[0]):
        if not wp.isfinite(errors[i]) or errors[i] > tolerance:
            rows[active] = i
            active += 1
    count[0] = active
    previous = int(0)
    for i in range(widths.shape[0]):
        conditions[i] = int(previous < active and active <= widths[i])
        previous = widths[i]


@wp.kernel
def pack_rows(
    source: wp.array2d[wp.float64],
    rows: wp.array[wp.int32],
    count: wp.array[wp.int32],
    target: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    value = wp.float64(0.0)
    if i < count[0]:
        value = source[rows[i], j]
    target[i, j] = value


@wp.kernel
def unpack_rows(
    source: wp.array2d[wp.float64],
    rows: wp.array[wp.int32],
    count: wp.array[wp.int32],
    target: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    if i < count[0]:
        target[rows[i], j] = source[i, j]


@wp.kernel
def record_batch(count: wp.array[wp.int32], usage: wp.array[wp.int32], bucket: int):
    wp.atomic_add(usage, bucket, 1)
    wp.atomic_add(usage, usage.shape[0] - 1, count[0])


class DeviceBucketedCG:
    """Wrap a BatchedCGPreconditioner and the solver's freshly recomputed row errors.

    Arrays keep fixed identities. Factors/working matrix are shared; packed
    buffers and upstream CG workspaces are distinct per bucket. Stats count
    scheduled vector iterations, including padding within the selected bucket.
    `usage[:-1]` counts calls per bucket; `usage[-1]` sums active RHSs. Omitted
    rows receive zero correction, but remain in the outer Ritz basis. This is
    residual-based soft locking, not a certificate of the requested spectrum.
    """

    requires_static_outer_capture = False

    def __init__(self, inverse, errors, tolerance, bucket_size=4):
        bucket_size = index(bucket_size)
        if bucket_size < 1:
            raise ValueError("bucket_size must be positive")
        if inverse.requires_static_outer_capture:
            raise ValueError("device CG buckets require upstream preallocated loop control")
        self.base = inverse
        self.width, self.n, self.device = inverse.width, inverse.n, inverse.device
        if (
            errors.shape != (self.width,)
            or errors.dtype != wp.float64
            or errors.device != self.device
        ):
            raise ValueError("errors must be FP64 rows on the inverse device")
        self.errors, self.tolerance = errors, float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        sizes = list(range(bucket_size, self.width, bucket_size)) + [self.width]
        self.widths = wp.array(sizes, dtype=wp.int32, device=self.device)
        self.rows = wp.empty(self.width, dtype=wp.int32, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.conditions = wp.zeros(len(sizes), dtype=wp.int32, device=self.device)
        self.flags = [self.conditions[i : i + 1] for i in range(len(sizes))]
        self.source = wp.empty((self.width, self.n), dtype=wp.float64, device=self.device)
        self.stats = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.usage = wp.zeros(len(sizes) + 1, dtype=wp.int32, device=self.device)
        self.states, self.rhs, self.outputs = [], [], []
        for width in sizes:
            state = inverse if width == self.width else inverse.for_width(width)
            rhs = wp.zeros((width, self.n), dtype=wp.float64, device=self.device)
            out = wp.empty_like(rhs)
            state.apply(rhs, out)  # Compile every state before conditional capture.
            self.states.append(state)
            self.rhs.append(rhs)
            self.outputs.append(out)
        wp.synchronize_device(self.device)
        self.reset_stats()

    def reset_stats(self):
        self.stats.zero_()
        self.usage.zero_()
        for state in self.states:
            state.reset_stats()

    def apply(self, x, y):
        if (
            x.shape != (self.width, self.n)
            or y.shape != x.shape
            or x.dtype != wp.float64
            or y.dtype != wp.float64
            or x.device != self.device
            or y.device != self.device
        ):
            raise ValueError("expected matching FP64 (width,n) device rows")
        wp.launch(
            select_rows,
            1,
            [self.errors, self.tolerance, self.rows, self.count, self.widths, self.conditions],
            device=self.device,
        )
        wp.copy(self.source, x)  # Also supports x/y aliasing.
        y.zero_()

        def branch(i):
            rhs, out, state = self.rhs[i], self.outputs[i], self.states[i]
            wp.launch(
                pack_rows, rhs.shape, [self.source, self.rows, self.count, rhs], device=self.device
            )
            state.reset_stats()
            state.apply(rhs, out)
            wp.launch(unpack_rows, out.shape, [out, self.rows, self.count, y], device=self.device)
            wp.launch(merge_statistics, 4, [state.stats, self.stats], device=self.device)
            wp.launch(record_batch, 1, [self.count, self.usage, i], device=self.device)

        if wp.get_stream(self.device).is_capturing:
            for i, condition in enumerate(self.flags):
                wp.capture_if(condition, lambda i=i: branch(i))
        else:
            # Metadata-only branch: no device value readback/host synchronization.
            branch(len(self.states) - 1)
