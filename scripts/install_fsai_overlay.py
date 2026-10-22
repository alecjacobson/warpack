"""Install three checksum-pinned upstream modules into an explicit Warp target.

First install the official Warp 1.18.0+cu12 wheel with pip --target, as documented
in docs/releases/v0.2.0.md. This never modifies the default Python installation.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMON_FILES = {
    "warp/_src/optim/linear.py": "1fe28335bafc86c6a35bdf962940ebff545d44f3e3818c76879f2e3919015e18",
    "warp/optim/linear.py": "4eac35c139553d1ce187fb33808a2dec6ccfe768fae174da65a66146a7e5666f",
}
PROFILES = {
    "v0.2.0": (
        "04427b997fe7ff07bc6ed1dde5ca9813496c6b55",
        "35c0f0cba8318fe74fa6b4b471679fe56a17c22f40331d3c89a2053bf62d9d07",
    ),
    "packed": (
        "61077a5c67ea9455cddb614cf68b9bd7e0df9379",
        "aa4f526521c35f87ddfc1610f18bbda395784b636a4a8c844bce8f40ef4ab8b2",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Explicit pip --target directory containing Warp")
    parser.add_argument("--profile", choices=PROFILES, default="v0.2.0")
    args = parser.parse_args()
    if not (args.target / "warp/__init__.py").is_file():
        parser.error("target must already contain the official Warp wheel")
    metadata = list(args.target.glob("warp_lang-*.dist-info/METADATA"))
    if len(metadata) != 1 or "\nVersion: 1.18.0+cu12\n" not in metadata[0].read_text():
        parser.error("target must contain exactly Warp 1.18.0+cu12")
    revision, fsai_digest = PROFILES[args.profile]
    files = COMMON_FILES | {"warp/_src/optim/fsai.py": fsai_digest}
    downloaded = {}
    for name, digest in files.items():
        url = f"https://raw.githubusercontent.com/alecjacobson/warp/{revision}/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Checksum mismatch: {name}; no files written")
        downloaded[name] = data
    for name, data in downloaded.items():
        destination = args.target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print(f"Verified and installed {name}")
    (args.target / "warpack-fsai-overlay.json").write_text(
        json.dumps(dict(profile=args.profile, revision=revision, sha256=files), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
