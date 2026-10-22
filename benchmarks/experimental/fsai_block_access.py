"""Isolate #141: load BSR scalar entries through a view rather than a whole block."""

import difflib
import hashlib
import importlib.util
import sys
from pathlib import Path

from benchmarks.experimental.fsai_current import load as load_current

PACKED_SHA256 = "aa4f526521c35f87ddfc1610f18bbda395784b636a4a8c844bce8f40ef4ab8b2"


def source_patch(source, layout="flat"):
    if hashlib.sha256(source.encode()).hexdigest() != PACKED_SHA256:
        raise ValueError("block-access experiment requires the qualified packed source")
    assert source.count("values: wp.array[block_type]") == 5
    if layout not in ("flat", "view"):
        raise ValueError("expected flat or view layout")
    annotation = (
        "wp.array[dtype]"
        if layout == "flat"
        else "wp.array(dtype=dtype, ndim=1 if block_type == dtype or block_type._shape_ == (1, 1) else 3)"
    )
    source = source.replace("values: wp.array[block_type]", "values: " + annotation)
    if layout == "view":
        source = source.replace("    if block_type == dtype:\n", "    if block_size == 1:\n", 1)
    source = source.replace(
        "dtype(values[slot][r % wp.static(block_size), c % wp.static(block_size)])",
        "values[slot * wp.static(block_size * block_size) + (r % wp.static(block_size)) * wp.static(block_size) + c % wp.static(block_size)]"
        if layout == "flat"
        else "values[slot, r % wp.static(block_size), c % wp.static(block_size)]",
    )
    source = source.replace(
        "A.columns, A.values,", "A.columns, A.values.view(A.scalar_type).flatten(),"
    )
    source = source.replace(
        "                A.values,\n", "                A.values.view(A.scalar_type).flatten(),\n"
    )
    if layout == "view":
        source = source.replace(
            "A.values.view(A.scalar_type).flatten()", "A.values.view(A.scalar_type)"
        )
    return source


def load(layout="flat"):
    original = Path(load_current(packed=True).__file__).read_text()
    source = source_patch(original, layout)
    suffix = "_view" if layout == "view" else ""
    path = Path(f"build/fsai_block_access{suffix}.py")
    path.write_text(source)
    path.with_suffix(".patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                source.splitlines(True),
                fromfile="a/warp/_src/optim/fsai.py",
                tofile="b/warp/_src/optim/fsai.py",
            )
        )
    )
    name = "warpack_experimental_fsai_block_access" + suffix
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
