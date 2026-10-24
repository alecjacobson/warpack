"""Explicit projection controls restore the public constructor, even on rejection."""

import json
import sys

import pytest

from benchmarks.experimental import compare_projection_whole as driver


@pytest.mark.parametrize("choice", driver.SETTINGS)
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_selected_control_and_restoration(monkeypatch, tmp_path, choice, compact, rejected):
    instances = []

    class DummyBlock:
        def __init__(self, **kwargs):
            self.options = kwargs
            instances.append(self)

    class DummyCandidate(DummyBlock):
        pass

    destination = tmp_path / "comparison.json"

    def compare():
        driver.warpack.BlockLOBPCG()
        destination.write_text(json.dumps({"args": {}, "accepted": not rejected}))
        if rejected:
            raise SystemExit(1)

    monkeypatch.setattr(driver.warpack, "BlockLOBPCG", DummyBlock)
    monkeypatch.setattr(driver, "ProjectionBlock", DummyCandidate)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    manifest = driver.compare_block.source_manifest
    argv = [
        "compare",
        "--projection",
        choice,
        "--public-api",
        "--fused-updates",
        "--out",
        str(destination),
        "--reference-cache",
        str(tmp_path / "cache"),
    ]
    if compact:
        argv.append("--compact-graphs")
    monkeypatch.setattr(sys, "argv", argv)
    if rejected:
        with pytest.raises(SystemExit) as caught:
            driver.main()
        assert caught.value.code == 1
    else:
        driver.main()
    assert driver.warpack.BlockLOBPCG is DummyBlock
    assert driver.compare_block.source_manifest is manifest and sys.argv is argv
    assert isinstance(instances[0], DummyCandidate) == (choice not in ("original", "default"))
    assert instances[0]._wide_projection_enabled == (choice == "default")
    assert instances[0].options == {"compact_projected_graphs": compact}
    actual = instances[0]._projection_rows
    assert actual == driver.SETTINGS[choice]
    report = json.loads(destination.read_text())
    assert report["args"]["projection"] == choice
    assert report["projection_rows"] == actual
    assert report["accepted"] == (not rejected)
    assert report["independent_reference_cache"] == []
