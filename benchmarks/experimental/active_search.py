"""Experimental device-bucketed block search spaces after residual soft locking.

All Ritz vectors remain in the trial space. Only residual/history search rows
whose fresh errors exceed tolerance are packed into smaller preallocated spaces.
`keep_history=True` retains all P rows and compacts only R.
Directions may reactivate after every Ritz update. Private block hooks are used;
this adapter is tied to this checkout and is not an installed API.
"""

import copy
from operator import index

import warp as wp

from benchmarks.experimental.active_cg import pack_rows
from benchmarks.experimental.cached_x_image import CachedXBlockLOBPCG, _ReuseOperator
from warpack.block import (
    Identity,
    normalize_gram,
    rotate,
    scale_rows,
    scales,
    symmetrize,
    whitening,
)
from warpack.solver import SymmetricEigensolver


@wp.kernel
def select_search(
    errors: wp.array[wp.float64],
    tolerance: wp.float64,
    rows: wp.array[wp.int32],
    count: wp.array[wp.int32],
    widths: wp.array[wp.int32],
    flags: wp.array[wp.int32],
):
    active = int(0)
    for i in range(errors.shape[0]):
        if not wp.isfinite(errors[i]) or errors[i] > tolerance:
            rows[active] = i
            active += 1
    count[0] = active
    previous = int(0)
    for i in range(widths.shape[0]):
        # A zero-search fixed step still performs a Ritz update in the X space.
        flags[i] = int((i == 0 and active == 0) or previous < active and active <= widths[i])
        previous = widths[i]


@wp.kernel
def record_search(usage: wp.array[wp.int32], bucket: int):
    wp.atomic_add(usage, bucket, 1)


class ActiveSearchBlockLOBPCG(CachedXBlockLOBPCG):
    """Keep the full X block; compact R/P on device into fixed graph branches."""

    def __init__(self, *args, search_bucket_size=4, keep_history=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_history = bool(keep_history)
        size = index(search_bucket_size)
        if size < 1:
            raise ValueError("search_bucket_size must be positive")
        sizes = list(range(size, self.b, size)) + [self.b]
        self.search_widths = wp.array(sizes, dtype=wp.int32, device=self.device)
        self.search_rows = wp.empty(self.b, dtype=wp.int32, device=self.device)
        self.search_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.search_flags = wp.zeros(len(sizes), dtype=wp.int32, device=self.device)
        self.search_usage = wp.zeros(len(sizes), dtype=wp.int32, device=self.device)
        self._search_conditions = [self.search_flags[i : i + 1] for i in range(len(sizes))]
        self._search_correction = wp.empty_like(self.r)
        self._search_buckets = [self._specialize(q) for q in sizes]

    def _specialize(self, width):
        branch = copy.copy(self)
        search_width = width + (self.b if self.keep_history else width)
        branch.m = self.b + search_width
        branch.s = self.s[: branch.m]
        branch.z = self.z[: branch.m]
        branch.az = self.az[: branch.m]
        branch.sx = branch.s[: self.b]
        branch.sw = branch.s[self.b : self.b + width]
        branch.sp = branch.s[self.b + width :]
        branch.su = branch.s[self.b :]
        branch.u1 = self.u1[:search_width]
        branch.u2 = self.u2[:search_width]
        branch.bu = self.bu[:search_width]
        branch.small = SymmetricEigensolver(Identity(branch.m, self.device), self.b, ncv=branch.m)
        branch.metric = SymmetricEigensolver(
            Identity(search_width, self.device), search_width // 2, ncv=search_width
        )
        branch.transform = wp.zeros(
            (search_width, search_width), dtype=wp.float64, device=self.device
        )
        branch.coeff = wp.zeros((self.b, branch.m), dtype=wp.float64, device=self.device)
        branch.projection_x = wp.zeros((search_width, self.b), dtype=wp.float64, device=self.device)
        branch.scale = wp.zeros(search_width, dtype=wp.float64, device=self.device)
        branch.active = self.active[: branch.m]
        branch.active_x = branch.active[: self.b]
        branch.active_u = branch.active[self.b :]
        if self.constraints is not None:
            branch.projection = self.projection[: branch.m]
            branch.projection_u = branch.projection[self.b :]
        branch._image_x = branch.az[: self.b]
        branch._image_scratch = branch.z[: self.b]
        branch.op = _ReuseOperator(branch, self._original_operator)
        return branch

    def whiten(self, x, out, normalize=False):
        # The public method currently sizes its launch as2*b. Here the search
        # width can be smaller; the same two-pass numerical procedure is kept.
        self.mass.apply(x, self.bu)
        self.gram(x, self.bu, self.metric.g, symmetric=True)
        if normalize:
            self.launch(scales, x.shape[0], [self.metric.g, self.scale, self.status])
            self.launch(scale_rows, x.shape, [x, self.scale])
            self.launch(normalize_gram, self.metric.g.shape, [self.metric.g, self.scale])
        else:
            self.launch(symmetrize, self.metric.g.shape, [self.metric.g])
        self.metric.which = 1
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

    def _step_bucket(self, branch, i):
        wp.copy(branch.sx, self.x)
        wp.launch(
            pack_rows,
            branch.sw.shape,
            [self._search_correction, self.search_rows, self.search_count, branch.sw],
            device=self.device,
        )
        if self.keep_history:
            wp.copy(branch.sp, self.p)
        else:
            wp.launch(
                pack_rows,
                branch.sp.shape,
                [self.p, self.search_rows, self.search_count, branch.sp],
                device=self.device,
            )
        wp.copy(branch._image_x, self.ax)
        branch._reuse_image = True
        try:
            branch.rayleigh_ritz()
        finally:
            branch._reuse_image = False
        wp.launch(record_search, 1, [self.search_usage, i], device=self.device)

    def step(self):
        wp.launch(
            select_search,
            1,
            [
                self.errors,
                self.tol,
                self.search_rows,
                self.search_count,
                self.search_widths,
                self.search_flags,
            ],
            device=self.device,
        )
        self.preconditioner.apply(self.r, self._search_correction)
        if wp.get_stream(self.device).is_capturing:
            for i, flag in enumerate(self._search_conditions):
                wp.capture_if(flag, lambda i=i: self._step_bucket(self._search_buckets[i], i))
        else:
            self._step_bucket(self._search_buckets[-1], len(self._search_buckets) - 1)

    def initialize(self, seed=42):
        super().initialize(seed)
        self.search_usage.zero_()

    def capture(self, max_iterations=100, adaptive=None, finalizer=None):
        self.solve(1)
        self.search_count.zero_()
        for i, branch in enumerate(self._search_buckets):
            self._step_bucket(branch, i)  # Warm every shape using zero search rows.
        wp.synchronize_device(self.device)
        return super().capture(max_iterations, adaptive=adaptive, finalizer=finalizer)
