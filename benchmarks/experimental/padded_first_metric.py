"""Certify the nonzero block of a normalized metric with exact zero padding.

The original Gram and its nullity remain intact. Auxiliary identity entries
only permit a full-rank certificate on a separate matrix; they never become
physical search directions. This is an experiment, not production dispatch.
"""

import warp as wp

from benchmarks.experimental.first_metric_certificate import FirstMetricCertificate


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


class PaddedFirstMetricCertificate:
    """Device-only certificate; callers must fall back when accepted is zero.

    Storage comprises auxiliary matrix, factor, work, inverse and congruence.
    Factorization work is reused for the first product only after factorization
    and inverse formation. No data needed by canonical fallback is overwritten.
    """

    def __init__(self, original, rank_tol=1e-12, *, storage=None):
        self.original, self.device = original, original.device
        n = original.shape[0]
        if storage is None:
            storage = tuple(wp.empty_like(original) for _ in range(5))
        if len(storage) != 5:
            raise ValueError("five scratch matrices required")
        auxiliary, factor, work, inverse, transformed = storage
        self.certificate = FirstMetricCertificate(
            auxiliary, rank_tol, storage=(factor, work, inverse, work, transformed)
        )
        if (
            original.shape != auxiliary.shape
            or original.dtype != auxiliary.dtype
            or original.device != auxiliary.device
        ):
            raise ValueError("matching original and auxiliary metrics required")
        self.active = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.candidate = wp.zeros_like(self.count)
        self.accepted = self.certificate.accepted
        self.inverse = self.certificate.inverse
        wp.load_module(module=prepare.module, device=self.device, block_dim=128)

    def prepare(self):
        self.accepted.zero_()
        wp.launch_tiled(
            prepare,
            dim=1,
            inputs=[self.original, self.certificate.g, self.active, self.count, self.candidate],
            block_dim=128,
            device=self.device,
        )

    def attempt(self):
        self.certificate.apply()
        wp.launch_tiled(
            certify_separation,
            dim=1,
            inputs=[self.inverse, self.active, self.accepted],
            block_dim=128,
            device=self.device,
        )

    def apply(self):
        self.prepare()
        wp.capture_if(self.candidate, self.attempt)
