"""Certified first search-metric whitening with exact padding and device fallback.

Auxiliary identity entries fill only proven zero padding in separate scratch.
They never become physical modes. Uncertain rank or failed outer state retains
canonical revelation on the original normalized Gram. Both whitening passes
remain; this helper shares dense scratch with the second pass.
"""

import warp as wp

from ._projected_cholesky import global_cholesky
from .block import gram, normalize_gram, rotate, scale_rows, scales, whitening


@wp.kernel(enable_backward=False)
def inverse_lower(
    factor: wp.array2d[wp.float64],
    inverse: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
):
    column = wp.tid()
    # Each thread owns one entire column. Explicitly initialize the upper
    # triangle as well: a fresh or previously failed replay cannot leak entries.
    for i in range(factor.shape[0]):
        value = wp.float64(0.0)
        if status[0] == 0 and i >= column:
            if i == column:
                value = wp.float64(1.0)
            for j in range(column, i):
                value -= factor[i, j] * inverse[j, column]
            value /= factor[i, i]
        inverse[i, column] = value


@wp.kernel(enable_backward=False)
def certify(
    g: wp.array2d[wp.float64],
    inverse: wp.array2d[wp.float64],
    transformed: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    rank_tol: wp.float64,
    accepted: wp.array[wp.int32],
    diagnostic: wp.array[wp.float64],
):
    block, lane = wp.tid()
    n = g.shape[0]
    bad = int(status[0] != 0)
    gn, rn, rc, tn, deviation = (
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )
    for i in range(lane, n, 128):
        gs, rs, cs, ts, ds = (
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
        )
        if inverse[i, i] <= wp.float64(0.0):
            bad = 1
        for j in range(n):
            gv = g[i, j]
            rv = inverse[i, j]
            cv = inverse[j, i]
            tv = transformed[i, j]
            if (
                not wp.isfinite(gv)
                or not wp.isfinite(rv)
                or not wp.isfinite(cv)
                or not wp.isfinite(tv)
            ):
                bad = 1
            if gv != g[j, i] or (i < j and rv != wp.float64(0.0)):
                bad = 1
            gs += wp.abs(gv)
            rs += wp.abs(rv)
            cs += wp.abs(cv)
            ts += wp.abs(tv)
            if i == j:
                tv -= wp.float64(1.0)
            ds += wp.abs(tv)
        gn, rn, rc = wp.max(gn, gs), wp.max(rn, rs), wp.max(rc, cs)
        tn, deviation = wp.max(tn, ts), wp.max(deviation, ds)
    bad = wp.tile_extract(wp.tile_max(wp.tile(bad)), 0)
    gn = wp.tile_extract(wp.tile_max(wp.tile(gn)), 0)
    rn = wp.tile_extract(wp.tile_max(wp.tile(rn)), 0)
    rc = wp.tile_extract(wp.tile_max(wp.tile(rc)), 0)
    tn = wp.tile_extract(wp.tile_max(wp.tile(tn)), 0)
    deviation = wp.tile_extract(wp.tile_max(wp.tile(deviation)), 0)
    if lane == 0:
        eps = wp.float64(2.220446049250313e-16)
        # gram uses padded256-entry dot chunks and atomic chunk accumulation.
        # Budget well above the full dot/reduction depth, using epsilon=2*u.
        padded = ((n + 255) // 256) * 256
        nu = eps * wp.float64(4 * (padded + n + 2))
        gamma = nu / (wp.float64(1.0) - nu)
        outward = wp.float64(1.0) + wp.float64(8.0) * eps
        inflation = outward / (wp.float64(1.0) - gamma)
        gn, rn, rc, tn = gn * inflation, rn * inflation, rc * inflation, tn * inflation
        product_norm = rn * gn * rc
        product_error = gamma * (wp.float64(2.0) + gamma) * product_norm
        # Include dot-product errors, absolute-row-sum and diagonal-subtraction
        # errors, plus an absolute allowance far above aggregate underflow.
        delta = (
            deviation + gamma * (tn + wp.float64(1.0)) + product_error + wp.float64(1e-280)
        ) * inflation
        lower = (wp.float64(1.0) - delta) / (rn * rc * outward)
        lower *= wp.float64(1.0) - wp.float64(8.0) * eps
        threshold = rank_tol * gn * outward
        valid = bad == 0 and wp.isfinite(rank_tol) and rank_tol >= wp.float64(0.0)
        valid = valid and wp.isfinite(delta) and wp.isfinite(lower) and wp.isfinite(gn)
        valid = valid and delta <= wp.float64(0.25) and lower > threshold
        accepted[0] = int(valid)
        diagnostic[0] = delta
        diagnostic[1] = lower
        diagnostic[2] = gn
        diagnostic[3] = product_error


@wp.kernel(enable_backward=False)
def prepare(
    original: wp.array2d[wp.float64],
    auxiliary: wp.array2d[wp.float64],
    active: wp.array[wp.int32],
    count: wp.array[wp.int32],
    candidate: wp.array[wp.int32],
):
    block, lane = wp.tid()
    n = original.shape[0]
    bad, rank = int(0), int(0)
    for i in range(lane, n, 128):
        d = original[i, i]
        present = d > wp.float64(0.0)
        if not wp.isfinite(d) or (
            d != wp.float64(0.0) and (d < wp.float64(0.5) or d > wp.float64(2.0))
        ):
            bad = 1
        active[i] = int(present)
        rank += int(present)
        for j in range(n):
            a = original[i, j]
            if not wp.isfinite(a) or a != original[j, i]:
                bad = 1
            if (not present or original[j, j] == wp.float64(0.0)) and a != wp.float64(0.0):
                bad = 1
            if i == j and not present:
                a = wp.float64(1.0)
            auxiliary[i, j] = a
    bad = wp.tile_extract(wp.tile_max(wp.tile(bad)), 0)
    rank = wp.tile_extract(wp.tile_sum(wp.tile(rank)), 0)
    if lane == 0:
        count[0] = rank
        candidate[0] = int(bad == 0 and rank > 0)


@wp.kernel(enable_backward=False)
def certify_separation(
    inverse: wp.array2d[wp.float64],
    active: wp.array[wp.int32],
    accepted: wp.array[wp.int32],
):
    block, lane = wp.tid()
    bad = int(0)
    n = inverse.shape[0]
    for i in range(lane, n, 128):
        for j in range(n):
            if active[i] == 0 or active[j] == 0:
                expected = wp.float64(0.0)
                if i == j:
                    expected = wp.float64(1.0)
                if inverse[i, j] != expected:
                    bad = 1
    bad = wp.tile_extract(wp.tile_max(wp.tile(bad)), 0)
    if lane == 0 and bad != 0:
        accepted[0] = 0


@wp.kernel(enable_backward=False)
def select_attempt(
    candidate: wp.array[wp.int32], outer_status: wp.array[wp.int32], flags: wp.array[wp.int32]
):
    # Failed scaling can turn a negative diagonal into apparent zero padding.
    # Preserve canonical error routing rather than certifying that masked input.
    flags[0] = int(candidate[0] != 0 and outer_status[0] == 0)
    flags[1] = 0
    flags[2] = 1 - flags[0]


@wp.kernel(enable_backward=False)
def choose(accepted: wp.array[wp.int32], flags: wp.array[wp.int32]):
    flags[1] = accepted[0]
    flags[2] = 1 - accepted[0]


@wp.kernel(enable_backward=False)
def record(flags: wp.array[wp.int32], usage: wp.array[wp.int32], fast: bool):
    if fast:
        usage[0] += 1
    else:
        usage[1] += 1
        usage[2] += 1 - flags[0]


class FirstWhitening:
    def __init__(self, owner, storage, usage):
        n = owner.metric.g.shape[0]
        self.device, self.usage = owner.device, usage
        self.matrix, self.factor, self.work = (a[:n, :n] for a in storage)
        self.transposed = self.matrix.transpose()
        self.inverse, self.transformed = owner.transform, owner.metric.v
        self.active = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.candidate = wp.zeros_like(self.count)
        self.factor_status = wp.zeros_like(self.count)
        self.accepted = wp.zeros_like(self.count)
        self.diagnostic = wp.empty(4, dtype=wp.float64, device=self.device)
        self.flags = wp.zeros(3, dtype=wp.int32, device=self.device)
        self.conditions = [self.flags[i : i + 1] for i in range(3)]
        self.threads = 256 if n <= 256 else 512
        wp.load_module(module=prepare.module, device=self.device, block_dim=128)
        wp.load_module(module=inverse_lower.module, device=self.device, block_dim=32)
        wp.load_module(module=global_cholesky.module, device=self.device, block_dim=self.threads)

    def apply(self, owner, x, out):
        owner.mass.apply(x, owner.bu)
        owner.gram(x, owner.bu, owner.metric.g, symmetric=True)
        owner.launch(scales, x.shape[0], [owner.metric.g, owner.scale, owner.status])
        owner.launch(scale_rows, x.shape, [x, owner.scale])
        owner.launch(normalize_gram, owner.metric.g.shape, [owner.metric.g, owner.scale])
        self.accepted.zero_()
        wp.launch_tiled(
            prepare,
            dim=1,
            inputs=[owner.metric.g, self.matrix, self.active, self.count, self.candidate],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            select_attempt,
            1,
            [self.candidate, owner.status, self.flags],
            block_dim=128,
            device=self.device,
        )

        def attempt():
            self.factor_status.zero_()
            wp.launch_tiled(
                global_cholesky,
                dim=1,
                inputs=[self.matrix, self.factor, self.factor_status, self.work],
                block_dim=self.threads,
                device=self.device,
            )
            wp.launch(
                inverse_lower,
                x.shape[0],
                [self.factor, self.inverse, self.factor_status],
                block_dim=32,
                device=self.device,
            )
            # Factor workspace is dead after inverse formation; reuse it for
            # the first product. Original metric.g remains untouched for fallback.
            gram(self.inverse, self.transposed, self.work)
            gram(self.work, self.inverse, self.transformed)
            wp.launch_tiled(
                certify,
                dim=1,
                inputs=[
                    self.matrix,
                    self.inverse,
                    self.transformed,
                    self.factor_status,
                    owner.rank_tol,
                    self.accepted,
                    self.diagnostic,
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch_tiled(
                certify_separation,
                dim=1,
                inputs=[self.inverse, self.active, self.accepted],
                block_dim=128,
                device=self.device,
            )
            wp.launch(choose, 1, [self.accepted, self.flags], block_dim=128, device=self.device)

        def fast():
            wp.copy(owner.active_u, self.active)
            wp.copy(owner.rank, self.count)
            rotate(x, self.inverse, out)
            wp.launch(record, 1, [self.flags, self.usage, True], block_dim=128, device=self.device)

        def canonical():
            owner.metric.which = 1
            owner.metric.diagonalize()
            owner.rank.zero_()
            owner.launch(
                whitening,
                owner.transform.shape,
                [
                    owner.metric.v,
                    owner.metric.ritz,
                    owner.metric.order,
                    owner.transform,
                    owner.active_u,
                    owner.rank,
                    owner.status,
                    owner.rank_tol,
                ],
            )
            rotate(x, owner.transform, out)
            wp.launch(record, 1, [self.flags, self.usage, False], block_dim=128, device=self.device)

        wp.capture_if(self.conditions[0], attempt)
        wp.capture_if(self.conditions[1], fast)
        wp.capture_if(self.conditions[2], canonical)
