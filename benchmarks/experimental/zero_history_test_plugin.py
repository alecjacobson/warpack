"""Run existing regression cases with the isolated zero-history warmup candidate."""

import pytest

from benchmarks.experimental.zero_history_warmup import ZeroHistorySearch
from warpack import _block_search


@pytest.fixture(autouse=True)
def candidate_search_warmup(monkeypatch):
    monkeypatch.setattr(_block_search, "ActiveSearch", ZeroHistorySearch)
