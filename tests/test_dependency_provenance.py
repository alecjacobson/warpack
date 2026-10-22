"""Same wheel version is insufficient to identify a patched numerical dependency."""

import json

import pytest

from benchmarks.dependency_provenance import fingerprint, signature


def test_same_version_different_module_bytes_and_location_independence(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_bytes(b"old implementation\n")
    second.write_bytes(first.read_bytes())
    a = fingerprint("1.18.0", {"fsai": first}, tmp_path)
    b = fingerprint("1.18.0", {"fsai": second}, tmp_path)
    assert a["files"]["fsai"]["path"] != b["files"]["fsai"]["path"]
    assert signature(a) == signature(b)
    second.write_bytes(b"new implementation\n")
    assert signature(a) != signature(fingerprint("1.18.0", {"fsai": second}, tmp_path))


def test_stale_manifest_cannot_masquerade_as_original_dependency(tmp_path):
    module = tmp_path / "fsai.py"
    module.write_bytes(b"original\n")
    before = fingerprint("1.18.0", {"fsai": module}, tmp_path)
    manifest = {"sha256": {"fsai": before["files"]["fsai"]["sha256"]}}
    (tmp_path / "warpack-fsai-overlay.json").write_text(json.dumps(manifest))
    verified = fingerprint("1.18.0", {"fsai": module}, tmp_path)
    assert verified["manifest_status"] == "verified"
    assert signature(before) == signature(verified)
    module.write_bytes(b"unrecorded replacement\n")
    stale = fingerprint("1.18.0", {"fsai": module}, tmp_path)
    assert stale["manifest_status"] == "mismatch"
    assert stale["mismatches"] == ["fsai"]
    with pytest.raises(ValueError, match="disagrees"):
        signature(stale)


def test_historical_record_needs_reviewed_dependency_baseline():
    with pytest.raises(ValueError, match="lacks loaded dependency"):
        signature(None)
