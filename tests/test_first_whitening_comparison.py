"""The installed comparison must retain an explicit canonical control."""

import json
import sys

import pytest

from benchmarks.experimental import compare_first_whitening as driver


@pytest.mark.parametrize("choice", ["canonical", "default"])
@pytest.mark.parametrize("compact", [False, True])
def test_control_dispatch_and_absent_usage(monkeypatch, tmp_path, choice, compact):
    instances = []

    class DummyBlock:
        _first_whitening_enabled = True

        def __init__(self, **kwargs):
            self._first_whitening_usage = None
            self.options = kwargs
            instances.append(self)

    destination = tmp_path / "control.json"

    def compare():
        driver.warpack.BlockLOBPCG()
        destination.write_text(json.dumps({"args": {}}))

    monkeypatch.setattr(driver.warpack, "BlockLOBPCG", DummyBlock)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    argv = ["compare", "--first-whitening", choice, "--public-api", "--out", str(destination)]
    if compact:
        argv.append("--compact-graphs")
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()
    assert driver.warpack.BlockLOBPCG is DummyBlock
    assert instances[0]._first_whitening_enabled == (choice == "default")
    assert instances[0].options == {"compact_projected_graphs": compact}
    report = json.loads(destination.read_text())
    assert report["args"] == {"first_whitening": choice, "compact_graphs": compact}
    assert report["first_whitening_usage_last_replay"] == []
    assert "Installed" in report["first_whitening_scope"]
