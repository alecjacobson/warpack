"""Read-only dependency identity for setup/reporting, never solver arithmetic."""

import hashlib
import inspect
import json
from pathlib import Path


def fingerprint(warp_version, modules, target):
    """Hash actual loaded sources; an optional installation manifest is only a claim."""
    files = {
        name: dict(
            path=str(Path(path).resolve()),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        )
        for name, path in modules.items()
    }
    manifest_path = Path(target) / "warpack-fsai-overlay.json"
    manifest, mismatches = None, []
    status = "absent"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        claimed = manifest.get("sha256", {})
        mismatches = sorted(
            name
            for name in set(claimed) | set(files)
            if claimed.get(name) != files.get(name, {}).get("sha256")
        )
        status = "mismatch" if mismatches else "verified"
    return dict(
        warp_version=warp_version,
        files=files,
        manifest=manifest,
        manifest_status=status,
        mismatches=mismatches,
    )


def warp_dependencies(fsai=None):
    import warp as wp
    from warp.optim import linear

    modules = {
        "warp/optim/linear.py": inspect.getfile(linear),
        "warp/_src/optim/linear.py": inspect.getfile(linear.LinearOperator),
    }
    constructor = fsai if fsai is not None else getattr(linear, "FSAI", None)
    if constructor is not None:
        modules["warp/_src/optim/fsai.py"] = inspect.getfile(constructor)
    return fingerprint(wp.__version__, modules, Path(wp.__file__).resolve().parent.parent)


def signature(provenance):
    """Compare content, not installation directories or unverified manifest claims."""
    if provenance is None:
        raise ValueError(
            "baseline lacks loaded dependency provenance; establish a reviewed baseline"
        )
    if provenance["manifest_status"] not in ("absent", "verified"):
        raise ValueError("dependency manifest disagrees with loaded modules")
    return dict(
        warp_version=provenance["warp_version"],
        files={k: v["sha256"] for k, v in provenance["files"].items()},
    )
