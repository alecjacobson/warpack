"""Install three checksum-pinned upstream modules into an explicit Warp target.

First install the official Warp 1.18.0+cu12 wheel with pip --target, as documented
in docs/releases/v0.2.0.md. This never modifies the default Python installation.
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "04427b997fe7ff07bc6ed1dde5ca9813496c6b55"
FILES = {
    "warp/_src/optim/linear.py": "1fe28335bafc86c6a35bdf962940ebff545d44f3e3818c76879f2e3919015e18",
    "warp/_src/optim/fsai.py": "35c0f0cba8318fe74fa6b4b471679fe56a17c22f40331d3c89a2053bf62d9d07",
    "warp/optim/linear.py": "4eac35c139553d1ce187fb33808a2dec6ccfe768fae174da65a66146a7e5666f",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Explicit pip --target directory containing Warp")
    args = parser.parse_args()
    if not (args.target / "warp/__init__.py").is_file():
        parser.error("target must already contain the official Warp wheel")
    metadata = list(args.target.glob("warp_lang-*.dist-info/METADATA"))
    if len(metadata) != 1 or "\nVersion: 1.18.0+cu12\n" not in metadata[0].read_text():
        parser.error("target must contain exactly Warp 1.18.0+cu12")
    downloaded = {}
    for name, digest in FILES.items():
        url = f"https://raw.githubusercontent.com/alecjacobson/warp/{REVISION}/{name}"
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


if __name__ == "__main__":
    main()
