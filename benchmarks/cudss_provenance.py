"""Identify an already-loaded cuDSS backend after measurement, without loading one."""

import ctypes as ct
import hashlib
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _loaded_library_path(library):
    # CDLL._name may be only a SONAME. Resolve the actual loaded symbol through
    # the dynamic loader instead of guessing from a pip package or search path.
    class DlInfo(ct.Structure):
        _fields_ = [
            ("filename", ct.c_char_p),
            ("base", ct.c_void_p),
            ("symbol", ct.c_char_p),
            ("address", ct.c_void_p),
        ]

    try:
        dladdr = ct.CDLL(None).dladdr
        dladdr.argtypes = [ct.c_void_p, ct.POINTER(DlInfo)]
        dladdr.restype = ct.c_int
        info = DlInfo()
        if dladdr(ct.cast(library.cudssGetProperty, ct.c_void_p), ct.byref(info)) and info.filename:
            return Path(os.fsdecode(info.filename)).resolve(strict=True)
    except (AttributeError, OSError):
        pass
    requested = Path(library._name)
    if requested.is_absolute():
        return requested.resolve(strict=True)
    raise ValueError("cannot identify the loaded cuDSS binary from its SONAME")


def _file(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def cudss_dependencies():
    """Return None if unused; otherwise record binary, binding and runtime identity.

    Call after setup/solve timers. Do not import the optional backend or call
    get_library here: that would move first-load costs out of a setup timer.
    Binary hashes describe the file backing the loaded symbol at report time;
    benchmark environments must keep their loaded libraries unchanged.
    """
    binding = sys.modules.get("warp_cudss._bindings")
    library = getattr(binding, "_lib", None)
    if library is None:
        return None
    runtime_version = []
    for property_id in range(3):
        value = ct.c_int()
        status = library.cudssGetProperty(property_id, ct.byref(value))
        if status:
            raise RuntimeError(f"cuDSS version property {property_id} failed with status {status}")
        runtime_version.append(value.value)
    package = Path(binding.__file__).resolve().parent
    try:
        installed_version = version("nvidia-cudss-cu12")
    except PackageNotFoundError:
        installed_version = None
    return dict(
        runtime_version=runtime_version,
        library=_file(_loaded_library_path(library)),
        binding_files={
            str(p.relative_to(package)): _file(p) for p in sorted(package.rglob("*.py"))
        },
        installed_package_version=installed_version,
        environment_at_reporting={
            key: os.environ.get(key)
            for key in ("CUDSS_LIBRARY_PATH", "CUDSS_JIT_CACHE_DIR", "CUDSS_JIT_DISABLE")
        },
        scope="Already-loaded symbol resolved after measurement; no backend preload. Package metadata and current override are context only, not the loaded binary identity. JIT settings describe the reporting environment; keep them fixed throughout each worker.",
    )


def cudss_signature(provenance):
    """Compare runtime/binary/source content, independent of installation paths."""
    if not provenance:
        raise ValueError("missing loaded cuDSS provenance; do not infer it from package metadata")
    return dict(
        runtime_version=provenance["runtime_version"],
        library_sha256=provenance["library"]["sha256"],
        binding_files={k: v["sha256"] for k, v in provenance["binding_files"].items()},
    )
