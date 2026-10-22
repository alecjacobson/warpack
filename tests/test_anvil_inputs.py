"""Exact matrix/data provenance must survive relocation and reject stale caches."""

import copy
import json
import shutil

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks import anvil_inputs as ai


def fixture_arrays():
    values = np.array([np.eye(3) * 2, np.eye(3) * -0.2, np.eye(3) * -0.2, np.eye(3) * 3])
    return dict(
        positions=np.arange(6, dtype=np.float64).reshape(2, 3),
        offsets=np.array([0, 2, 4], np.int32),
        columns=np.array([0, 1, 0, 1], np.int32),
        stiffness=values.copy(),
        normalized=values.copy(),
        mass=np.ones(2),
        rigid=np.zeros((6, 6)),
    )


def test_identity_tracks_values_structure_shape_dtype_and_ignores_sparse_format():
    a = sp.csr_matrix([[2.0, 0, -1], [0, 4, 0], [-1, 0, 3]])
    assert ai.matrix_identity(a) == ai.matrix_identity(a.tocsc())
    changed = a.copy()
    changed.data[0] += 0.1
    assert ai.matrix_identity(a) != ai.matrix_identity(changed)
    changed = a.copy()
    changed.indices[1] = 1
    assert ai.matrix_identity(a) != ai.matrix_identity(changed)
    x = np.arange(6, dtype=np.float64)
    assert ai.array_identity(x) != ai.array_identity(x.reshape(2, 3))
    assert ai.array_identity(x) != ai.array_identity(x.astype(np.float32))


def test_cache_is_portable_verified_and_never_silently_replaced(tmp_path):
    configuration = dict(mesh="bytes", assembly="code", material="parameters")
    path = tmp_path / "first.npz"
    manifest = ai.save(path, fixture_arrays(), configuration)
    moved = tmp_path / "second.npz"
    shutil.copyfile(path, moved)
    assert ai.load(path, configuration)[1] == ai.load(moved, configuration)[1] == manifest
    with pytest.raises(FileExistsError):
        ai.save(path, fixture_arrays(), configuration)
    with pytest.raises(ValueError, match="mesh/assembly/material"):
        ai.load(path, configuration | dict(material="changed"))
    arrays = fixture_arrays()
    arrays["mass"][0] *= 2
    np.savez(moved, **arrays, manifest=np.array(json.dumps(manifest)))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ai.load(moved, configuration)


def test_common_inputs_separate_method_starts_and_reject_missing_history():
    a = fixture_arrays()
    record = ai.inputs(
        dict(K="k", M="m"), dict(values="v", vectors="x"), np.ones(6), ai.host_matrix(a)
    )
    changed = copy.deepcopy(record)
    changed["method"]["initial"]["sha256"] = "other"
    assert ai.signature(record, common_only=True) == ai.signature(changed, common_only=True)
    assert ai.signature(record) != ai.signature(changed)
    for bad in (None, {}, dict(schema=1, common={}, method={})):
        with pytest.raises(ValueError, match="input provenance"):
            ai.signature(bad)


def test_failed_cache_write_does_not_leave_a_valid_looking_fixture(tmp_path, monkeypatch):
    path = tmp_path / "failed.npz"

    def fail(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("disk full")

    monkeypatch.setattr(ai.np, "savez", fail)
    with pytest.raises(OSError, match="disk full"):
        ai.save(path, fixture_arrays(), {"mesh": "test"})
    assert not path.exists()


@pytest.mark.parametrize("shift", [1.0, 100.0])
def test_cached_device_and_cpu_shifted_matrices_are_identical(shift):
    a = fixture_arrays()
    original, shifted = ai.shifted_device(a, shift)
    expected = ai.host_matrix(a)
    assert ai.matrix_identity(ai.host_device_matrix(original)) == ai.matrix_identity(expected)
    assert ai.matrix_identity(ai.host_device_matrix(shifted)) == ai.matrix_identity(
        expected + shift * sp.eye(6)
    )
