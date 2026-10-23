"""Benchmark-only guarded first whitening with unchanged canonical fallback."""

import warp as wp

from benchmarks.experimental.first_metric_certificate import FirstMetricCertificate
from warpack import BlockLOBPCG
from warpack.block import normalize_gram, rotate, scale_rows, scales, whitening


@wp.kernel(enable_backward=False)
def candidate(g: wp.array2d[wp.float64], flags: wp.array[wp.int32]):
    block, lane = wp.tid()
    bad = int(0)
    for i in range(lane, g.shape[0], 128):
        diagonal = g[i, i]
        # Enforce the normalized-input scale used in the underflow allowance,
        # and cheaply exclude padding before doing cubic factor/inverse work.
        if not wp.isfinite(diagonal) or diagonal < wp.float64(0.5) or diagonal > wp.float64(2.0):
            bad = 1
    bad = wp.tile_extract(wp.tile_max(wp.tile(bad)), 0)
    if lane == 0:
        flags[0] = int(bad == 0)
        flags[1] = 0
        flags[2] = int(bad != 0)


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
        matrix, factor, work = (a[:n, :n] for a in storage)
        # Original metric.g is retained for fallback. metric.v is reset by
        # canonical diagonalization before it is used as eigenvectors.
        self.certificate = FirstMetricCertificate(
            owner.metric.g,
            owner.rank_tol,
            storage=(factor, work, owner.transform, matrix, owner.metric.v),
        )
        self.flags = wp.zeros(3, dtype=wp.int32, device=owner.device)
        self.conditions = [self.flags[i : i + 1] for i in range(3)]
        self.usage = usage
        wp.load_module(module=candidate.module, device=owner.device, block_dim=128)

    def prepare_candidate(self, owner):
        wp.launch_tiled(
            candidate,
            dim=1,
            inputs=[owner.metric.g, self.flags],
            block_dim=128,
            device=owner.device,
        )

    def attempt(self, owner):
        self.certificate.apply()
        wp.launch(choose, 1, [self.certificate.accepted, self.flags], device=owner.device)

    def fast(self, owner, x, out):
        owner.active_u.fill_(1)
        owner.rank.fill_(x.shape[0])
        rotate(x, owner.transform, out)
        wp.launch(record, 1, [self.flags, self.usage, True], device=owner.device)

    def apply(self, owner, x, out):
        owner.mass.apply(x, owner.bu)
        owner.gram(x, owner.bu, owner.metric.g, symmetric=True)
        owner.launch(scales, x.shape[0], [owner.metric.g, owner.scale, owner.status])
        owner.launch(scale_rows, x.shape, [x, owner.scale])
        owner.launch(normalize_gram, owner.metric.g.shape, [owner.metric.g, owner.scale])
        self.prepare_candidate(owner)

        def attempt():
            self.attempt(owner)

        def fast():
            self.fast(owner, x, out)

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
            wp.launch(record, 1, [self.flags, self.usage, False], device=owner.device)

        wp.capture_if(self.conditions[0], attempt)
        wp.capture_if(self.conditions[1], fast)
        wp.capture_if(self.conditions[2], canonical)


class FirstWhiteningBlock(BlockLOBPCG):
    # Keep this prototype independent of the installed first-pass dispatcher.
    _first_whitening_enabled = False
    _first_metric_minimum = 96
    _first_whitener_type = FirstWhitening

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        branches = [self] + ([] if self._search is None else self._search.buckets)
        eligible = []
        for branch in branches:
            branch._first_whitener = None
            if self.device.is_cuda and self._first_metric_minimum <= branch.metric.m <= 512:
                eligible.append(branch)
        self._first_whitening_usage = wp.zeros(3, dtype=wp.int32, device=self.device)
        self._first_whitening_storage = None
        if eligible:
            size = max(b.metric.m for b in eligible)
            storage = self._second_whitening_storage
            if storage is None or storage[0].shape[0] < size:
                storage = tuple(
                    wp.empty((size, size), dtype=wp.float64, device=self.device) for _ in range(3)
                )
            self._first_whitening_storage = storage
            for branch in eligible:
                branch._first_whitener = self._first_whitener_type(
                    branch, storage, self._first_whitening_usage
                )

    def whiten(self, x, out, normalize=False):
        if (
            normalize
            and self._first_whitener is not None
            and wp.get_stream(self.device).is_capturing
        ):
            self._first_whitener.apply(self, x, out)
        else:
            super().whiten(x, out, normalize)
