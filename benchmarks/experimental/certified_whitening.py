"""Explicit size controls for the qualified internal second whitening path."""

from warpack import BlockLOBPCG


class CertifiedWhiteningBlock(BlockLOBPCG):
    # Benchmark-only: force all CUDA sizes to compare the original prototype
    # scope with the production guard (wide Ada metrics only).
    _metric_whitening_minimum = 2
    _metric_whitening_arches = None

    @property
    def whitening_usage(self):
        return self._second_whitening_usage


class CanonicalWhiteningBlock(BlockLOBPCG):
    _metric_whitening_minimum = 513
