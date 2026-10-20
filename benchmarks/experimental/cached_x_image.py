"""Compatibility constructor for the public block operator-image reuse option."""

from warpack import BlockLOBPCG


class CachedXBlockLOBPCG(BlockLOBPCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reuse_operator_images=True, **kwargs)
