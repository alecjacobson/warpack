"""Reject wheels/installations whose Python modules differ from current core source."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def audit(wheel, source, installed=None):
    wheel, source = Path(wheel), Path(source)
    files = {
        p.relative_to(source).as_posix(): p.read_bytes() for p in (source / "warpack").rglob("*.py")
    }
    if "warpack/__init__.py" not in files:
        raise ValueError("source must contain the warpack package")
    expected = set(files)
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel archive paths")
        actual = {name for name in names if name.endswith(".py")}
        if actual != expected:
            raise ValueError(
                f"wheel module inventory: extra={sorted(actual - expected)}, "
                f"missing={sorted(expected - actual)}"
            )
        for name, content in files.items():
            if archive.read(name) != content:
                raise ValueError(f"wheel source bytes differ: {name}")
    if installed is not None:
        installed = Path(installed)
        actual = {p.relative_to(installed).as_posix() for p in installed.rglob("*.py")}
        if actual != expected:
            raise ValueError(
                f"installed module inventory: extra={sorted(actual - expected)}, "
                f"missing={sorted(expected - actual)}"
            )
        for name, content in files.items():
            if (installed / name).read_bytes() != content:
                raise ValueError(f"installed source bytes differ: {name}")
    return dict(
        accepted=True,
        wheel=str(wheel.resolve()),
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        installed=None if installed is None else str(installed.resolve()),
        modules=len(files),
        source_sha256={
            name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--installed", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; retain previous audit evidence")
    try:
        report = audit(args.wheel, args.source, args.installed)
    except (ValueError, OSError, BadZipFile) as error:
        report = dict(accepted=False, error=str(error), wheel=str(args.wheel.resolve()))
        if args.wheel.is_file():
            report["sha256"] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit(report["error"])
    print(f"Verified {report['modules']} core modules; wheel SHA256 {report['sha256']}")


if __name__ == "__main__":
    main()
