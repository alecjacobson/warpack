"""Installed/original controls are explicit and survive rejected comparisons."""

import json
import sys

import pytest

from benchmarks.experimental import compare_metric_qr as driver


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_original_installed_controls_and_restoration(monkeypatch, tmp_path, installed, rejected):
    instances = []

    class DummyBlock:
        def __init__(self, **kwargs):
            instances.append(self)
            self._ritz_qr_usage = None

    destination = tmp_path / "control.json"

    def compare():
        driver.warpack.BlockLOBPCG()
        destination.write_text(json.dumps({"args": {}, "accepted": not rejected}))
        if rejected:
            raise SystemExit(1)

    monkeypatch.setattr(driver.warpack, "BlockLOBPCG", DummyBlock)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    original_manifest, original_prepare = (
        driver.compare_block.source_manifest,
        driver.compare_block.prepare,
    )
    argv = [
        "compare",
        "--metric-defect",
        "0",
        "--public-api",
        "--out",
        str(destination),
        "--reference-cache",
        str(tmp_path / "cache"),
    ]
    if installed:
        argv.append("--installed-default")
    monkeypatch.setattr(sys, "argv", argv)
    if rejected:
        with pytest.raises(SystemExit) as caught:
            driver.main()
        assert caught.value.code == 1
    else:
        driver.main()
    assert instances[0]._ritz_qr_enabled == installed
    assert driver.warpack.BlockLOBPCG is DummyBlock
    assert driver.compare_block.source_manifest is original_manifest
    assert driver.compare_block.prepare is original_prepare and sys.argv is argv
    report = json.loads(destination.read_text())
    assert report["args"]["installed_default"] == installed
    assert report["accepted"] == (not rejected)
