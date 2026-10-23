"""Gram benchmark controls restore the public class even after rejection."""

import json
import sys

import pytest

from benchmarks.experimental import compare_gram_whole as driver


@pytest.mark.parametrize("tile,threads", [(0, 128), (8, 64), (16, 128), (32, 256)])
@pytest.mark.parametrize("rejected", [False, True])
def test_controls_and_rejected_report(monkeypatch, tmp_path, tile, threads, rejected):
    instances = []

    class DummyBlock:
        def __init__(self, **kwargs):
            self.options = kwargs
            instances.append(self)

    class DummyPrototype(DummyBlock):
        pass

    destination = tmp_path / "control.json"

    def compare():
        driver.warpack.BlockLOBPCG()
        destination.write_text(json.dumps({"args": {}, "accepted": not rejected}))
        if rejected:
            raise SystemExit(1)

    monkeypatch.setattr(driver.warpack, "BlockLOBPCG", DummyBlock)
    monkeypatch.setattr(driver, "GramTileBlock", DummyPrototype)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    original_manifest = driver.compare_block.source_manifest
    argv = [
        "compare",
        "--gram-tile",
        str(tile),
        "--gram-threads",
        str(threads),
        "--public-api",
        "--compact-graphs",
        "--out",
        str(destination),
        "--reference-cache",
        str(tmp_path / "cache"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    if rejected:
        with pytest.raises(SystemExit) as caught:
            driver.main()
        assert caught.value.code == 1
    else:
        driver.main()
    assert driver.warpack.BlockLOBPCG is DummyBlock
    assert driver.compare_block.source_manifest is original_manifest and sys.argv is argv
    assert isinstance(instances[0], DummyPrototype) == bool(tile)
    assert instances[0]._gram_output_tile == tile and instances[0]._gram_threads == threads
    assert instances[0].options == {"compact_projected_graphs": True}
    report = json.loads(destination.read_text())
    assert report["args"]["gram_tile"] == tile and report["args"]["gram_threads"] == threads
    assert report["args"]["compact_graphs"] and report["accepted"] == (not rejected)
    assert report["independent_reference_cache"] == []
