"""CPU reference reuse must preserve provenance, equations and atomic publication."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from benchmarks.reference_cache import InvalidReferenceCache, cached_reference


def compute():
    return {"values": np.arange(1.0, 5.0), "vectors": np.eye(4)}


def validate(arrays):
    w, v = arrays["values"], arrays["vectors"]
    assert w.shape == (4,) and v.shape == (4, 4)
    np.testing.assert_allclose(np.diag(np.arange(1.0, 5.0)) @ v, v * w, atol=1e-14)
    np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-14)


def test_reuse_does_not_recompute_and_validates_each_read(tmp_path):
    checks = []

    def check(arrays):
        validate(arrays)
        checks.append(True)

    first, p = cached_reference(tmp_path, {"source": "a", "count": 4}, compute, check)
    before = len(checks)

    def forbidden():
        raise AssertionError("unexpected recomputation")

    second, q = cached_reference(tmp_path, {"count": 4, "source": "a"}, forbidden, check)
    assert len(checks) == before + 1
    assert p["status"] == "created" and q["status"] == "verified_reuse"
    assert p["manifest_sha256"] == q["manifest_sha256"]
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "matrix_values",
        "matrix_structure",
        "dtype",
        "count",
        "guard",
        "algorithm",
        "dependency",
    ],
)
def test_changed_provenance_gets_a_distinct_entry(tmp_path, field):
    request = {
        name: "initial"
        for name in [
            "source",
            "matrix_values",
            "matrix_structure",
            "dtype",
            "count",
            "guard",
            "algorithm",
            "dependency",
        ]
    }
    _, first = cached_reference(tmp_path, request, compute, validate)
    _, second = cached_reference(tmp_path, request | {field: "changed"}, compute, validate)
    assert first["key"] != second["key"] and second["status"] == "created"


@pytest.mark.parametrize("kind", ["payload", "request", "array_hash", "partial"])
def test_corrupt_entries_fail_closed_without_recomputation(tmp_path, kind):
    _, evidence = cached_reference(tmp_path, {"source": "a"}, compute, validate)
    entry = tmp_path / evidence["key"]
    if kind == "payload":
        (entry / "arrays.npz").write_bytes(b"not an archive")
    elif kind == "partial":
        (entry / "manifest.json").unlink()
    else:
        p = entry / "manifest.json"
        content = json.loads(p.read_text())
        if kind == "request":
            content["request"]["source"] = "wrong"
        else:
            content["arrays"]["vectors"]["sha256"] = "wrong"
        p.write_text(json.dumps(content))

    def forbidden():
        raise AssertionError("must not regenerate corrupt entries")

    with pytest.raises(InvalidReferenceCache):
        cached_reference(tmp_path, {"source": "a"}, forbidden, validate)


@pytest.mark.parametrize("kind", ["object", "nonfinite", "equations"])
def test_invalid_creation_is_not_published(tmp_path, kind):
    def broken():
        data = compute()
        if kind == "object":
            data["values"] = np.array([object()], dtype=object)
        elif kind == "nonfinite":
            data["values"][0] = np.nan
        else:
            data["values"][0] = 99
        return data

    with pytest.raises((ValueError, AssertionError)):
        cached_reference(tmp_path, {"source": "a"}, broken, validate)
    assert not any(p.is_dir() for p in tmp_path.iterdir())
    _, evidence = cached_reference(tmp_path, {"source": "a"}, compute, validate)
    assert evidence["status"] == "created"


def test_concurrent_creation_publishes_one_verified_entry(tmp_path):
    calls = []

    def tracked():
        calls.append(True)
        return compute()

    def worker(_):
        return cached_reference(tmp_path, {"source": "a"}, tracked, validate)[1]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    assert len(calls) == 1
    assert sum(r["status"] == "created" for r in results) == 1
    assert len({r["manifest_sha256"] for r in results}) == 1
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1


def test_correct_checksums_do_not_replace_numerical_validation(tmp_path):
    import hashlib

    from benchmarks.anvil_inputs import array_identity

    _, evidence = cached_reference(tmp_path, {"source": "a"}, compute, validate)
    entry = tmp_path / evidence["key"]
    wrong = compute()
    wrong["values"][0] = 99.0
    np.savez(entry / "arrays.npz", **wrong)
    manifest = json.loads((entry / "manifest.json").read_text())
    manifest["payload_sha256"] = hashlib.sha256((entry / "arrays.npz").read_bytes()).hexdigest()
    manifest["arrays"] = {name: array_identity(a) for name, a in wrong.items()}
    (entry / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InvalidReferenceCache, match="numerical reference"):
        cached_reference(tmp_path, {"source": "a"}, compute, validate)


def _process_cache_worker(root):
    return cached_reference(root, {"source": "process"}, compute, validate)[1]


def test_processes_do_not_read_partial_publication(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_process_cache_worker, [str(tmp_path)] * 3))
    assert sum(r["status"] == "created" for r in results) == 1
    assert len({r["manifest_sha256"] for r in results}) == 1


def test_loading_object_archive_is_rejected_even_with_valid_file_hash(tmp_path):
    import hashlib

    _, evidence = cached_reference(tmp_path, {"source": "a"}, compute, validate)
    entry = tmp_path / evidence["key"]
    wrong = compute()
    wrong["values"] = np.array(["pickle payload"], dtype=object)
    np.savez(entry / "arrays.npz", **wrong)
    manifest = json.loads((entry / "manifest.json").read_text())
    manifest["payload_sha256"] = hashlib.sha256((entry / "arrays.npz").read_bytes()).hexdigest()
    (entry / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(InvalidReferenceCache):
        cached_reference(tmp_path, {"source": "a"}, compute, validate)
