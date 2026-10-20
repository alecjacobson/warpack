"""Preallocated device-selected block search workspaces (internal)."""

import copy
import weakref

import warp as wp

from ._active_cg import pack_rows
from .block import Identity
from .solver import SymmetricEigensolver


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


class ActiveSearch:
    def __init__(self, owner, size, keep_history):
        self.owner = weakref.proxy(owner)
        self.keep_history = bool(keep_history)
        owner.keep_history = self.keep_history
        sizes = list(range(size, owner.b, size)) + [owner.b]
        self.widths = wp.array(sizes, dtype=wp.int32, device=owner.device)
        self.rows = wp.empty(owner.b, dtype=wp.int32, device=owner.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=owner.device)
        self.flags = wp.zeros(len(sizes), dtype=wp.int32, device=owner.device)
        self.usage = wp.zeros(len(sizes), dtype=wp.int32, device=owner.device)
        self.conditions = [self.flags[i : i + 1] for i in range(len(sizes))]
        self.correction = wp.empty_like(owner.r)
        self.buckets = [self._specialize(q) for q in sizes]

    def _specialize(self, width):
        owner = self.owner
        branch = copy.copy(owner)
        search_width = width + (owner.b if owner.keep_history else width)
        branch.m = owner.b + search_width
        branch.s = owner.s[: branch.m]
        branch.z = owner.z[: branch.m]
        branch.az = owner.az[: branch.m]
        branch.sx = branch.s[: owner.b]
        branch.sw = branch.s[owner.b : owner.b + width]
        branch.sp = branch.s[owner.b + width :]
        branch.su = branch.s[owner.b :]
        branch.u1 = owner.u1[:search_width]
        branch.u2 = owner.u2[:search_width]
        branch.bu = owner.bu[:search_width]
        branch.small = SymmetricEigensolver(Identity(branch.m, owner.device), owner.b, ncv=branch.m)
        branch.metric = SymmetricEigensolver(
            Identity(search_width, owner.device), search_width // 2, ncv=search_width
        )
        branch.transform = wp.zeros(
            (search_width, search_width), dtype=wp.float64, device=owner.device
        )
        branch.coeff = wp.zeros((owner.b, branch.m), dtype=wp.float64, device=owner.device)
        branch.projection_x = wp.zeros(
            (search_width, owner.b), dtype=wp.float64, device=owner.device
        )
        branch.scale = wp.zeros(search_width, dtype=wp.float64, device=owner.device)
        branch.active = owner.active[: branch.m]
        branch.active_x = branch.active[: owner.b]
        branch.active_u = branch.active[owner.b :]
        if owner.constraints is not None:
            branch.projection = owner.projection[: branch.m]
            branch.projection_u = branch.projection[owner.b :]
        branch._image_x = branch.az[: owner.b]
        branch._image_scratch = branch.z[: owner.b]
        return branch

    def step_bucket(self, branch, i):
        owner = self.owner
        wp.copy(branch.sx, owner.x)
        wp.launch(
            pack_rows,
            branch.sw.shape,
            [self.correction, self.rows, self.count, branch.sw],
            device=owner.device,
        )
        if self.keep_history:
            wp.copy(branch.sp, owner.p)
        else:
            wp.launch(
                pack_rows,
                branch.sp.shape,
                [owner.p, self.rows, self.count, branch.sp],
                device=owner.device,
            )
        wp.copy(branch._image_x, owner.ax)
        branch._image_reuse_active = True
        try:
            branch.rayleigh_ritz()
        finally:
            branch._image_reuse_active = False
        wp.launch(record_search, 1, [self.usage, i], device=owner.device)

    def step(self):
        owner = self.owner
        wp.launch(
            select_search,
            1,
            [owner.errors, owner.tol, self.rows, self.count, self.widths, self.flags],
            device=owner.device,
        )
        owner.preconditioner.apply(owner.r, self.correction)
        if wp.get_stream(owner.device).is_capturing:
            for i, flag in enumerate(self.conditions):
                wp.capture_if(flag, lambda i=i: self.step_bucket(self.buckets[i], i))
        else:
            self.step_bucket(self.buckets[-1], len(self.buckets) - 1)

    def warm(self):
        self.count.zero_()
        for i, branch in enumerate(self.buckets):
            self.step_bucket(branch, i)
