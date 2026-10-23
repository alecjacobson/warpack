"""Internal certified second metric whitening with canonical device fallback.

A fresh conservative spectral bound must certify all retained directions
after the first rank-revealing pass before Cholesky QR is used.
"""

import warp as wp

from . import kernels as K
from ._projected_cholesky import global_cholesky
from .block import rotate, symmetrize, whitening


@wp.kernel(enable_backward=False)
def certificate(
    g: wp.array2d[wp.float64],
    active: wp.array[wp.int32],
    rank_tol: wp.float64,
    flags: wp.array[wp.int32],
):
    block, lane = wp.tid()
    bad, count = int(0), int(0)
    deviation = wp.float64(0.0)
    for i in range(lane, g.shape[0], 128):
        row_error = wp.float64(0.0)
        if active[i] == 1:
            count += 1
        elif active[i] != 0:
            bad = 1
        for j in range(g.shape[1]):
            value = g[i, j]
            if not wp.isfinite(value):
                bad = 1
            if active[i] == 0 or active[j] == 0:
                if value != wp.float64(0.0):
                    bad = 1
            else:
                if i == j:
                    value -= wp.float64(1.0)
                row_error += wp.abs(value)
        deviation = wp.max(deviation, row_error)
    bad = wp.tile_extract(wp.tile_max(wp.tile(bad)), 0)
    deviation = wp.tile_extract(wp.tile_max(wp.tile(deviation)), 0)
    count = wp.tile_extract(wp.tile_sum(wp.tile(count)), 0)
    # Inflate the computed row-sum bound to cover summation/subtraction
    # roundoff. A borderline certificate must take the canonical fallback.
    deviation += (
        wp.float64(4.0 * 2.220446049250313e-16)
        * wp.float64(g.shape[0])
        * (wp.float64(1.0) + deviation)
    )
    if lane == 0:
        valid = bad == 0 and deviation <= wp.float64(0.25)
        valid = valid and (wp.float64(1.0) - deviation > rank_tol * (wp.float64(1.0) + deviation))
        flags[0] = int(valid)  # Candidate factorization.
        flags[1] = 0  # Successful factorization / fast application.
        flags[2] = int(not valid)  # Canonical fallback.
        flags[3] = count


@wp.kernel(enable_backward=False)
def fill_inactive_identity(
    original: wp.array2d[wp.float64], active: wp.array[wp.int32], work: wp.array2d[wp.float64]
):
    i, j = wp.tid()
    value = original[i, j]
    if i == j and active[i] == 0:
        value = wp.float64(1.0)
    work[i, j] = value


@wp.kernel(enable_backward=False)
def choose_factor(status: wp.array[wp.int32], flags: wp.array[wp.int32]):
    flags[1] = int(status[0] == 0)
    flags[2] = int(status[0] != 0)


@wp.kernel(enable_backward=False)
def accept_rank(flags: wp.array[wp.int32], rank: wp.array[wp.int32], usage: wp.array[wp.int32]):
    rank[0] = flags[3]
    usage[0] += 1


@wp.kernel(enable_backward=False)
def record_fallback(usage: wp.array[wp.int32]):
    usage[1] += 1


class SecondWhitening:
    def __init__(self, size, device, usage, storage):
        self.device, self.usage = device, usage
        # Search branches execute serially, so all share the owner's storage.
        self.matrix, self.factor, self.work = (array[:size, :size] for array in storage)
        self.flags = wp.zeros(4, dtype=wp.int32, device=device)
        self.conditions = [self.flags[i : i + 1] for i in range(3)]
        self.factor_status = wp.zeros(1, dtype=wp.int32, device=device)
        self.threads = 256 if size <= 256 else 512
        wp.load_module(module=certificate.module, device=device, block_dim=128)
        wp.load_module(module=global_cholesky.module, device=device, block_dim=self.threads)

    def apply(self, owner, x, out):
        owner.mass.apply(x, owner.bu)
        owner.gram(x, owner.bu, owner.metric.g, symmetric=True)
        owner.launch(symmetrize, owner.metric.g.shape, [owner.metric.g])
        wp.launch_tiled(
            certificate,
            dim=1,
            inputs=[owner.metric.g, owner.active_u, owner.rank_tol, self.flags],
            block_dim=128,
            device=self.device,
        )

        def factor():
            wp.launch(
                fill_inactive_identity,
                self.matrix.shape,
                [owner.metric.g, owner.active_u, self.matrix],
                device=self.device,
            )
            self.factor_status.zero_()
            wp.launch_tiled(
                global_cholesky,
                dim=1,
                inputs=[self.matrix, self.factor, self.factor_status, self.work],
                block_dim=self.threads,
                device=self.device,
            )
            wp.launch(choose_factor, 1, [self.factor_status, self.flags], device=self.device)

        def fast():
            wp.copy(out, x)
            owner.launch(K.triangular, owner.n, [out, self.factor])
            wp.launch(accept_rank, 1, [self.flags, owner.rank, self.usage], device=self.device)

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
            wp.launch(record_fallback, 1, [self.usage], device=self.device)

        wp.capture_if(self.conditions[0], factor)
        wp.capture_if(self.conditions[1], fast)
        wp.capture_if(self.conditions[2], canonical)
