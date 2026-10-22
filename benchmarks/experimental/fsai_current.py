"""Load a checksum-verified FSAI revision for qualification, without changing Warp."""

import difflib
import hashlib
import importlib.util
import sys
from pathlib import Path
from urllib.request import urlopen

REVISION = "429dccd0a133973c0809bdd974d81fcfc744919f"
SHA256 = "584e0fb277ae78d0f6e382b551898831680275396aacbcfb10e948961e141c8c"


def load(path="build/current_fsai.py", download=False, *, packed=False):
    path = Path(path)
    if not path.exists() and download:
        url = f"https://raw.githubusercontent.com/alecjacobson/warp/{REVISION}/warp/_src/optim/fsai.py"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("current FSAI checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("current FSAI checksum mismatch")
    name = "warpack_qualified_fsai_current"
    if packed:
        from benchmarks.experimental.fsai_packed_storage import patched_source

        source = path.read_text()
        updated = patched_source(source, base_sha256=SHA256)
        target = path.with_name("current_fsai_packed.py")
        target.write_text(updated)
        target.with_suffix(".patch").write_text(
            "".join(
                difflib.unified_diff(
                    source.splitlines(True),
                    updated.splitlines(True),
                    fromfile="a/warp/_src/optim/fsai.py",
                    tofile="b/warp/_src/optim/fsai.py",
                )
            )
        )
        path = target
        name += "_packed"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
