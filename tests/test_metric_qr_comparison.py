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
    original_reference = driver.compare_block.cupy_trials
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
    assert driver.compare_block.cupy_trials is original_reference
    report = json.loads(destination.read_text())
    assert report["args"]["installed_default"] == installed
    assert report["accepted"] == (not rejected)


@pytest.mark.parametrize("reference_error", [False, True])
def test_reference_quality_never_reads_or_attributes_warp_qr_counters(
    monkeypatch, tmp_path, reference_error
):
    class DummyBlock:
        def __init__(self, **kwargs):
            self._ritz_qr_usage = None

    destination = tmp_path / "reference.json"
    field = "metric_qr_reuse_and_fallback"

    def reference(check):
        assert field not in check()
        if reference_error:
            raise ValueError("reference failure")

    def compare():
        driver.warpack.BlockLOBPCG()
        check = driver.compare_block.prepare()[-1]
        assert check()[field] is None
        if reference_error:
            with pytest.raises(ValueError, match="reference failure"):
                driver.compare_block.cupy_trials(check)
        else:
            driver.compare_block.cupy_trials(check)
        assert check()[field] is None
        destination.write_text(json.dumps({"args": {}}))

    monkeypatch.setattr(driver.warpack, "BlockLOBPCG", DummyBlock)
    monkeypatch.setattr(driver.compare_block, "prepare", lambda: (lambda: {"accepted": True},))
    monkeypatch.setattr(driver.compare_block, "cupy_trials", reference)
    monkeypatch.setattr(driver.compare_block, "main", compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--metric-defect",
            "0",
            "--installed-default",
            "--cupy",
            "--public-api",
            "--out",
            str(destination),
            "--reference-cache",
            str(tmp_path / "cache"),
        ],
    )
    driver.main()
    assert driver.compare_block.cupy_trials is reference
    assert driver.warpack.BlockLOBPCG is DummyBlock


def test_reference_requires_installed_policy(monkeypatch, tmp_path):
    original = driver.warpack.BlockLOBPCG
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--metric-defect",
            "1e-13",
            "--cupy",
            "--public-api",
            "--out",
            str(tmp_path / "bad.json"),
            "--reference-cache",
            str(tmp_path / "cache"),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        driver.main()
    assert caught.value.code == 2 and driver.warpack.BlockLOBPCG is original
