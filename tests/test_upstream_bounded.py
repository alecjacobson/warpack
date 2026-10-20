"""A bounded inventory cannot report success without any input cases."""

import json
import struct
import sys
from types import SimpleNamespace

import pytest

from benchmarks.upstream import bounded


def fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    case = dict(id="Example2_0", family="Example2", selection=0)
    (folder / "Example2.jsonl").write_text(json.dumps(case) + "\n")
    (folder / "Example2_0.a").write_bytes(struct.pack("qq", 3, 3))
    return case


@pytest.mark.parametrize("present", [False, True])
def test_empty_fixture_inventory_is_an_error(tmp_path, monkeypatch, present, capsys):
    monkeypatch.setattr(bounded, "ROOT", tmp_path)
    if present:
        (tmp_path / "build/spectra-fixtures").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["bounded", "spectra", "--fresh"])
    with pytest.raises(SystemExit) as error:
        bounded.main()
    assert error.value.code == 2
    assert "No matching exported fixtures" in capsys.readouterr().err
    assert not (tmp_path / "results/upstream_spectra_warp.json").exists()


def test_empty_selection_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bounded, "ROOT", tmp_path)
    fixture(tmp_path / "build/spectra-fixtures")
    monkeypatch.setattr(sys, "argv", ["bounded", "spectra", "--selection", "4"])
    with pytest.raises(SystemExit) as error:
        bounded.main()
    assert error.value.code == 2
    assert "No matching exported fixtures" in capsys.readouterr().err


def test_explicit_accuracy_worker_and_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(bounded, "ROOT", tmp_path)
    folder = tmp_path / "exported"
    case = fixture(folder)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--fixtures") + 1] == str(folder)
        assert "--accuracy-only" in command
        assert kwargs["timeout"] == 30
        path = command[command.index("--out") + 1]
        with open(path, "w") as f:
            json.dump(dict(cases=[dict(**case, outcome="passed")]), f)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bounded.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bounded", "spectra", "--fixtures", str(folder), "--accuracy-only", "--out", "run.json"],
    )
    bounded.main()
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["accuracy_only"]
    assert report["cases"][0]["outcome"] == "passed"
    bounded.main()
    assert len(calls) == 1  # Valid complete resume does not rerun or fail.


@pytest.mark.parametrize("option", [["--accuracy-only"], ["--fixtures", "unused"]])
def test_spectra_only_options_reject_benchmark_suite(monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["bounded", "benchmarks", *option])
    with pytest.raises(SystemExit) as error:
        bounded.main()
    assert error.value.code == 2
