"""First-whitening experiment that certifies exact padding without changing rank."""

import warp as wp

from benchmarks.experimental.first_whitening import (
    FirstWhitening,
    FirstWhiteningBlock,
    choose,
    record,
)
from benchmarks.experimental.padded_first_metric import PaddedFirstMetricCertificate
from warpack.block import rotate


@wp.kernel(enable_backward=False)
def select_attempt(candidate: wp.array[wp.int32], flags: wp.array[wp.int32]):
    flags[0] = candidate[0]
    flags[1] = 0
    flags[2] = 1 - candidate[0]


class PaddedFirstWhitening(FirstWhitening):
    def __init__(self, owner, storage, usage):
        n = owner.metric.g.shape[0]
        matrix, factor, work = (a[:n, :n] for a in storage)
        self.certificate = PaddedFirstMetricCertificate(
            owner.metric.g,
            owner.rank_tol,
            storage=(matrix, factor, work, owner.transform, owner.metric.v),
        )
        self.flags = wp.zeros(3, dtype=wp.int32, device=owner.device)
        self.conditions = [self.flags[i : i + 1] for i in range(3)]
        self.usage = usage
        wp.load_module(module=select_attempt.module, device=owner.device)

    def prepare_candidate(self, owner):
        self.certificate.prepare()
        wp.launch(select_attempt, 1, [self.certificate.candidate, self.flags], device=owner.device)

    def attempt(self, owner):
        self.certificate.attempt()
        wp.launch(choose, 1, [self.certificate.accepted, self.flags], device=owner.device)

    def fast(self, owner, x, out):
        wp.copy(owner.active_u, self.certificate.active)
        wp.copy(owner.rank, self.certificate.count)
        rotate(x, owner.transform, out)
        wp.launch(record, 1, [self.flags, self.usage, True], device=owner.device)


class PaddedFirstWhiteningBlock(FirstWhiteningBlock):
    _first_whitener_type = PaddedFirstWhitening
