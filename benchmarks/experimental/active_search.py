"""Compatibility constructor for the public device-selected block search option."""

from warpack import BlockLOBPCG
from warpack._block_search import select_search as select_search


class ActiveSearchBlockLOBPCG(BlockLOBPCG):
    def __init__(self, *args, search_bucket_size=4, keep_history=False, **kwargs):
        super().__init__(
            *args,
            reuse_operator_images=True,
            search_bucket_size=search_bucket_size,
            keep_history=keep_history,
            **kwargs,
        )
