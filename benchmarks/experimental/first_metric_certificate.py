"""Experimental full-rank certificate for normalized first search metrics (#171).

No dispatch change. Successful Cholesky alone is insufficient: certify the
actual computed congruence and a conservative lower spectral bound. The caller
must retain canonical rank revelation whenever `accepted` is zero.
"""

import warp as wp

from warpack._projected_cholesky import global_cholesky
from warpack.block import gram


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


class FirstMetricCertificate:
    """Preallocated component experiment; accepts a fixed normalized FP64 Gram."""

    def __init__(self, g, rank_tol=1e-12):
        n = g.shape[0]
        if g.ndim != 2 or g.shape != (n, n) or g.dtype != wp.float64:
            raise ValueError("a square FP64 Gram is required")
        if not g.device.is_cuda or not 2 <= n <= 512:
            raise ValueError("CUDA dimension2..512 required")
        self.g, self.gt, self.rank_tol = g, g.transpose(), float(rank_tol)
        self.device, self.n = g.device, n
        self.factor, self.work, self.inverse, self.image, self.transformed = (
            wp.empty((n, n), dtype=wp.float64, device=self.device) for _ in range(5)
        )
        self.status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.accepted = wp.zeros_like(self.status)
        self.diagnostic = wp.empty(4, dtype=wp.float64, device=self.device)

    def apply(self):
        self.status.zero_()
        wp.launch_tiled(
            global_cholesky,
            dim=1,
            inputs=[self.g, self.factor, self.status, self.work],
            block_dim=256 if self.n <= 256 else 512,
            device=self.device,
        )
        wp.launch(
            inverse_lower,
            self.n,
            [self.factor, self.inverse, self.status],
            block_dim=32,
            device=self.device,
        )
        gram(self.inverse, self.gt, self.image)
        gram(self.image, self.inverse, self.transformed)
        wp.launch_tiled(
            certify,
            dim=1,
            inputs=[
                self.g,
                self.inverse,
                self.transformed,
                self.status,
                self.rank_tol,
                self.accepted,
                self.diagnostic,
            ],
            block_dim=128,
            device=self.device,
        )
