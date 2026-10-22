"""Cached independent references cannot silently cross matrices or hide corruption."""

import json

import numpy as np
import pytest

from benchmarks.applications import spectral_surface_reference as cache
from benchmarks.applications.spectral_surface import surface_pencil


def test_cached_reference_is_immutable_and_does_not_recompute(tmp_path, monkeypatch):
    p = surface_pencil(3)
    values, vectors, record = cache.reference(p, 1, tmp_path)
    assert not record["cache_hit"]
    path = next(tmp_path.glob("*.npz"))
    original = path.read_bytes()

    def fail(*_):
        raise AssertionError("reference recomputed on a cache hit")

    monkeypatch.setattr(cache, "compute", fail)
    w, v, hit = cache.reference(p, 1, tmp_path)
    assert hit["cache_hit"]
    np.testing.assert_array_equal(w, values)
    np.testing.assert_array_equal(v, vectors)
    assert path.read_bytes() == original


def test_changed_metric_has_distinct_reference_identity(tmp_path):
    first = cache.reference(surface_pencil(3), 1, tmp_path)
    second = cache.reference(surface_pencil(3, alpha=10), 1, tmp_path)
    assert first[2]["key"] != second[2]["key"]
    assert not second[2]["cache_hit"]
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert abs(first[0][0] - second[0][0]) > 1


@pytest.mark.parametrize("failure", ["checksum", "identity", "resigned-invalid-basis"])
def test_bad_cache_fails_instead_of_silently_replacing_it(tmp_path, failure):
    p = surface_pencil(3)
    cache.reference(p, 1, tmp_path)
    path = next(tmp_path.glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data}
    metadata = json.loads(str(arrays["metadata"]))
    if failure == "checksum":
        arrays["values"][0] += 0.5
    elif failure == "identity":
        metadata["identity"]["method"] = "unverified"
    else:
        arrays["vectors"][:, 0] = 1
        metadata["vectors_sha256"] = cache.array_hash(arrays["vectors"])
    arrays["metadata"] = json.dumps(metadata)
    with path.open("wb") as output:
        np.savez(output, **arrays)
    corrupted = path.read_bytes()
    with pytest.raises(ValueError, match="reference"):
        cache.reference(p, 1, tmp_path)
    assert path.read_bytes() == corrupted


def test_failed_publication_cleans_temporary_file(tmp_path, monkeypatch):
    def fail(*_):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(cache.os, "link", fail)
    with pytest.raises(OSError, match="publication"):
        cache.reference(surface_pencil(3), 1, tmp_path)
    assert list(tmp_path.iterdir()) == []
