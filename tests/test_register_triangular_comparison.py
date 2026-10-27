"""Performance controls must disable only the selected installed specialization."""

import json
import sys

import pytest

from benchmarks.experimental import compare_register_whole as driver


@pytest.mark.parametrize("choice", ["canonical", "default", "register"])
@pytest.mark.parametrize("compact", [False, True])
def test_explicit_control_and_restoration(monkeypatch, tmp_path, choice, compact):
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
        destination.write_text(json.dumps({"args": {}}))

    monkeypatch.setattr(driver.warpack, "BlockLOBPCG", DummyBlock)
    monkeypatch.setattr(driver, "WideRegisterTriangular", DummyPrototype)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    original_manifest = driver.compare_block.source_manifest
    argv = [
        "compare",
        "--triangular",
        choice,
        "--public-api",
        "--out",
        str(destination),
        "--reference-cache",
        str(tmp_path / "cache"),
    ]
    if compact:
        argv.append("--compact-graphs")
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()
    assert driver.warpack.BlockLOBPCG is DummyBlock
    assert driver.compare_block.source_manifest is original_manifest and sys.argv is argv
    assert instances[0]._register_triangular_enabled == (choice == "default")
    assert isinstance(instances[0], DummyPrototype) == (choice == "register")
    assert instances[0].options == {"compact_projected_graphs": compact}
    report = json.loads(destination.read_text())
    assert report["args"]["triangular"] == choice
    assert report["args"]["compact_graphs"] == compact
    assert report["independent_reference_cache"] == []
