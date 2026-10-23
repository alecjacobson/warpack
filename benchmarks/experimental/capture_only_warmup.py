"""Record each active-search branch directly, without an extra numerical warmup.

Benchmark-only: conditional CUDA graphs require drivers new enough to load
kernels during capture. BlockLOBPCG still warms its initial solve and synchronizes
before recording the complete outer graph. No selectable branch is omitted.
"""

from warpack._block_search import ActiveSearch


class CaptureOnlySearch(ActiveSearch):
    def warm(self):
        pass
