"""Explicit rotation controls restore the public constructor, even on rejection."""

import json
import sys

import pytest

from benchmarks.experimental import compare_paired_rotation_whole as driver


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
    monkeypatch.setattr(driver, "PairedRotationBlock", DummyCandidate)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    manifest = driver.compare_block.source_manifest
    argv = [
        "compare",
        "--rotation",
        choice,
        "--public-api",
        "--paired-rotation",
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
    assert instances[0]._wide_paired_rotation_enabled == (choice == "default")
    assert instances[0].options == {"compact_projected_graphs": compact}
    actual = tuple(
        getattr(instances[0], "_rotation_" + key) for key in ["rows", "columns", "inner", "threads"]
    )
    assert actual == driver.SETTINGS[choice]
    report = json.loads(destination.read_text())
    assert report["args"]["rotation"] == choice
    assert report["rotation_tile"] == list(actual)
    assert report["accepted"] == (not rejected)
    assert report["independent_reference_cache"] == []
