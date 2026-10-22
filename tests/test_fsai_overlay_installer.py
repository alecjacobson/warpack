"""Dependency provenance and all-downloads-before-writes installer guarantees."""

import hashlib
import io
import json
import sys

import pytest

from scripts import install_fsai_overlay as installer


def target(tmp_path):
    (tmp_path / "warp").mkdir()
    (tmp_path / "warp/__init__.py").write_text("# official target\n")
    metadata = tmp_path / "warp_lang-1.18.0+cu12.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: warp-lang\nVersion: 1.18.0+cu12\n")
    return tmp_path


def fixture(monkeypatch, tmp_path):
    root = target(tmp_path)
    payloads = {name: (name + "\n").encode() for name in installer.COMMON_FILES}
    payloads["warp/_src/optim/fsai.py"] = b"verified fsai\n"
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    monkeypatch.setattr(
        installer, "COMMON_FILES", {k: v for k, v in hashes.items() if not k.endswith("fsai.py")}
    )
    monkeypatch.setattr(
        installer, "PROFILES", {"test": ("immutable-revision", hashes["warp/_src/optim/fsai.py"])}
    )
    monkeypatch.setattr(sys, "argv", ["install", str(root), "--profile", "test"])
    for name in payloads:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"previous module\n")
    return root, payloads, hashes


def test_last_checksum_failure_never_partially_overwrites(monkeypatch, tmp_path):
    root, payloads, _ = fixture(monkeypatch, tmp_path)
    manifest = root / "warpack-fsai-overlay.json"
    manifest.write_text('"previous profile"\n')

    def download(url, timeout):
        path = url.split("/immutable-revision/")[1]
        return io.BytesIO(b"bad download" if path.endswith("fsai.py") else payloads[path])

    monkeypatch.setattr(installer, "urlopen", download)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        installer.main()
    assert all((root / name).read_bytes() == b"previous module\n" for name in payloads)
    assert manifest.read_text() == '"previous profile"\n'


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default-current"])
def test_verified_profile_records_exact_provenance(monkeypatch, tmp_path, explicit):
    root, payloads, hashes = fixture(monkeypatch, tmp_path)
    profile = "test" if explicit else "current"
    if not explicit:
        monkeypatch.setattr(installer, "PROFILES", {"current": installer.PROFILES["test"]})
        monkeypatch.setattr(sys, "argv", ["install", str(root)])
    monkeypatch.setattr(
        installer,
        "urlopen",
        lambda url, timeout: io.BytesIO(payloads[url.split("/immutable-revision/")[1]]),
    )
    installer.main()
    assert all((root / name).read_bytes() == data for name, data in payloads.items())
    assert json.loads((root / "warpack-fsai-overlay.json").read_text()) == dict(
        profile=profile, revision="immutable-revision", sha256=hashes
    )


def test_wrong_wheel_version_rejected_before_download(monkeypatch, tmp_path):
    root = target(tmp_path)
    next(root.glob("*.dist-info/METADATA")).write_text("Version: 1.15.0\n")
    monkeypatch.setattr(sys, "argv", ["install", str(root)])

    def download(*args, **kwargs):
        pytest.fail("download must not happen with the wrong wheel")

    monkeypatch.setattr(installer, "urlopen", download)
    with pytest.raises(SystemExit):
        installer.main()
