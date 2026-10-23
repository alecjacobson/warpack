"""Optional verified CPU reference storage, entirely outside solver timing/core.

The caller supplies a complete provenance request and validates the reference
against the current matrices on every use. A corrupt entry fails closed; this
module never replaces an existing entry or uses pickle serialization.
"""

import fcntl
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.anvil_inputs import array_identity


class InvalidReferenceCache(ValueError):
    """An existing entry is incomplete, corrupt, stale or numerically invalid."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_arrays(arrays):
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("reference must be a nonempty array mapping")
    result = {}
    for name, array in arrays.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("reference array names must be simple identifiers")
        a = np.ascontiguousarray(array)
        if a.dtype.kind not in "biufc" or not np.all(np.isfinite(a)):
            raise ValueError("reference arrays must be finite numeric data")
        result[name] = a
    return result


def read_entry(entry, request, key):
    try:
        raw = (entry / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if manifest["schema"] != 1 or manifest["request"] != request or manifest["key"] != key:
            raise ValueError("reference provenance mismatch")
        payload = (entry / "arrays.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
            raise ValueError("reference payload checksum mismatch")
        # Read the same verified bytes rather than reopening a mutable path.
        import io

        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            arrays = checked_arrays({name: data[name] for name in data.files})
        if {name: array_identity(a) for name, a in arrays.items()} != manifest["arrays"]:
            raise ValueError("reference array identity mismatch")
        return arrays, hashlib.sha256(raw).hexdigest()
    except Exception as error:
        raise InvalidReferenceCache(f"Invalid reference cache entry {entry}: {error}") from error


def cached_reference(root, request, compute, validate):
    """Return arrays and provenance, with serialized creation and atomic publication.

    `validate(arrays)` must raise on invalid shape, capacity, spectrum or original
    equations. It runs for newly computed data and every verified cache hit.
    Backend factors/preconditioners must never be supplied through this cache.
    """
    request = json.loads(encoded(request))
    key = hashlib.sha256(encoded(request)).hexdigest()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        created = not entry.exists()
        if created:
            arrays = checked_arrays(compute())
            validate(arrays)
            staging = Path(tempfile.mkdtemp(prefix="." + key + "-", dir=root))
            try:
                np.savez(staging / "arrays.npz", **arrays)
                manifest = dict(
                    schema=1,
                    key=key,
                    request=request,
                    arrays={name: array_identity(a) for name, a in arrays.items()},
                    payload_sha256=hashlib.sha256(
                        (staging / "arrays.npz").read_bytes()
                    ).hexdigest(),
                )
                (staging / "manifest.json").write_bytes(encoded(manifest))
                # Readers take the same per-key lock and cannot observe staging.
                staging.rename(entry)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        arrays, manifest_sha256 = read_entry(entry, request, key)
        try:
            validate(arrays)
        except Exception as error:
            raise InvalidReferenceCache(f"Invalid numerical reference {entry}: {error}") from error
    return arrays, dict(
        status="created" if created else "verified_reuse",
        key=key,
        path=str(entry),
        manifest_sha256=manifest_sha256,
        request=request,
        arrays={name: array_identity(a) for name, a in arrays.items()},
    )
