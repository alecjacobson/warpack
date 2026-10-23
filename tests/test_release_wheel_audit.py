"""Packaging identity checks must reject stale, missing and modified source."""

import json
import sys
from zipfile import ZipFile

import pytest

from scripts import audit_release_wheel as checker


def fixture(tmp_path, entries=None):
    source, installed = tmp_path / "source", tmp_path / "installed"
    files = {"warpack/__init__.py": b"", "warpack/solver.py": b"value = 1\n"}
    for directory in (source, installed):
        for name, content in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    wheel = tmp_path / "package.whl"
    with ZipFile(wheel, "w") as archive:
        for name, content in (files if entries is None else entries).items():
            archive.writestr(name, content)
        archive.writestr("package.dist-info/METADATA", "Name: package\n")
    return wheel, source, installed


def test_exact_inventory_and_bytes(tmp_path):
    wheel, source, installed = fixture(tmp_path)
    report = checker.audit(wheel, source, installed)
    assert report["accepted"] and report["modules"] == 2
    assert set(report["source_sha256"]) == {"warpack/__init__.py", "warpack/solver.py"}


@pytest.mark.parametrize("change", ["stale", "missing", "modified", "outside_core", "duplicate"])
def test_rejected_wheel(tmp_path, change):
    wheel, source, _ = fixture(tmp_path)
    files = {"warpack/__init__.py": b"", "warpack/solver.py": b"value = 1\n"}
    if change == "missing":
        files.pop("warpack/solver.py")
    elif change == "modified":
        files["warpack/solver.py"] = b"value = 2\n"
    elif change in ("stale", "outside_core"):
        files["warpack/skinning.py" if change == "stale" else "benchmarks/example.py"] = b""
    with ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        if change == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("warpack/solver.py", b"value = 1\n")
    with pytest.raises(ValueError, match="differ|inventory|duplicate"):
        checker.audit(wheel, source)


@pytest.mark.parametrize("change", ["stale", "missing", "modified"])
def test_rejected_installed_tree(tmp_path, change):
    wheel, source, installed = fixture(tmp_path)
    if change == "stale":
        (installed / "warpack/fem.py").write_text("")
    elif change == "missing":
        (installed / "warpack/solver.py").unlink()
    else:
        (installed / "warpack/solver.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="installed"):
        checker.audit(wheel, source, installed)


def test_cli_preserves_rejected_evidence(tmp_path, monkeypatch):
    wheel, source, installed = fixture(tmp_path)
    (installed / "warpack/fem.py").write_text("")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            str(wheel),
            "--source",
            str(source),
            "--installed",
            str(installed),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        checker.main()
    first = output.read_bytes()
    assert json.loads(first)["accepted"] is False
    with pytest.raises(SystemExit):
        checker.main()
    assert output.read_bytes() == first


def test_corrupt_archive_records_rejection(tmp_path, monkeypatch):
    wheel, source, _ = fixture(tmp_path)
    wheel.write_bytes(b"not a zip archive")
    output = tmp_path / "corrupt.json"
    monkeypatch.setattr(
        sys, "argv", ["audit", str(wheel), "--source", str(source), "--out", str(output)]
    )
    with pytest.raises(SystemExit):
        checker.main()
    report = json.loads(output.read_text())
    assert report["accepted"] is False and len(report["sha256"]) == 64
