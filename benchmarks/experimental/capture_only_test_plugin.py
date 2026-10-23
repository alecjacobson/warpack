"""Exercise existing regressions with direct capture of every search branch."""

import pytest

from benchmarks.experimental.capture_only_warmup import CaptureOnlySearch
from warpack import _block_search


@pytest.fixture(autouse=True)
def candidate_search_warmup(monkeypatch):
    monkeypatch.setattr(_block_search, "ActiveSearch", CaptureOnlySearch)
