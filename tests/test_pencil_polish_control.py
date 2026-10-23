"""Explicit polish controls retain rejected evidence and restore driver state."""

import json
import sys

import pytest

from benchmarks.experimental import pencil_polish_control as control


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("accepted", [False, True])
def test_control_restores_driver_and_keeps_rejection(monkeypatch, tmp_path, steps, accepted):
    calls = []
    destination = tmp_path / "result.json"

    def setup(*args, **kwargs):
        calls.append(kwargs)

    def diagnostic():
        control.driver.warp_setup("case", 64, 1e-9)
        destination.write_text(json.dumps(dict(args={}, accepted=accepted)))
        if not accepted:
            raise SystemExit(1)

    monkeypatch.setattr(control.driver, "warp_setup", setup)
    monkeypatch.setattr(control.driver, "main", diagnostic)
    manifest = control.driver.source_manifest
    argv = ["compare", "--polish-steps", str(steps), "--out", str(destination)]
    monkeypatch.setattr(sys, "argv", argv)
    if accepted:
        control.main()
    else:
        with pytest.raises(SystemExit) as error:
            control.main()
        assert error.value.code == 1
    assert control.driver.warp_setup is setup
    assert control.driver.source_manifest is manifest and sys.argv is argv
    assert calls == [{"polish": steps}]
    report = json.loads(destination.read_text())
    assert report["accepted"] is accepted and report["args"]["polish_steps"] == steps
    assert "Original eigenvalues are preserved" in report["protocol"]
