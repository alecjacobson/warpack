"""Procedural timing gates must compare actual coefficients and common starts."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks import block_inputs
from benchmarks.anvil_inputs import array_identity
from benchmarks.experimental import compare_block
from benchmarks.upstream.adapters import matrix


def reference(n=12, k=3):
    return block_inputs.reference_identity(
        SimpleNamespace(
            expected=np.arange(1, k + 1, dtype=float),
            requested_basis=np.eye(n)[:, :k],
            null_basis=np.empty((n, 0)),
            nullity=0,
            k=k,
            require_positive_guard=False,
        ),
        guard=np.array([k + 1.0]),
    )


def test_default_starts_unchanged_and_widths_share_exact_prefix():
    legacy = np.random.default_rng(42).normal(size=(100, 12))
    np.testing.assert_array_equal(block_inputs.initial_block(100, 12), legacy)
    narrow = block_inputs.initial_block(100, 8, pool_width=12)
    np.testing.assert_array_equal(narrow, legacy[:, :8])
    assert narrow.flags.c_contiguous
    assert array_identity(narrow) == array_identity(legacy[:, :8])
    assert not np.array_equal(block_inputs.initial_block(100, 8), narrow)


@pytest.mark.parametrize("args", [(0, 3), (12, 0), (12, 3, 2), (-1, 3)])
def test_invalid_start_pool_dimensions(args):
    with pytest.raises(ValueError):
        block_inputs.initial_block(*args)


def test_empty_null_basis_hash_preserves_existing_nonempty_identities():
    empty = np.empty((12, 0))
    record = array_identity(empty)
    assert record["shape"] == [12, 0]
    assert record["sha256"] == hashlib.sha256(b"").hexdigest()
    values = np.arange(48, dtype=np.float64).reshape(12, 4)
    assert (
        array_identity(values)["sha256"] == hashlib.sha256(memoryview(values).cast("B")).hexdigest()
    )
    assert reference()["null_basis"] == record


def test_canonical_sparse_identity_and_explicit_start_pool():
    a = sp.diags(np.arange(1, 13, dtype=float), format="csr")
    pool = block_inputs.initial_pool(12, 4, 6)
    start = np.ascontiguousarray(pool[:, :4])
    first = block_inputs.identities(a, None, a + sp.eye(12), reference(), start, 3, pool)
    other = block_inputs.identities(
        a.tobsr(blocksize=(3, 3)), None, (a + sp.eye(12)).tocsc(), reference(), start, 3, pool
    )
    assert block_inputs.signature(first) == block_inputs.signature(other)
    assert first["operators"]["mass"] == dict(kind="identity", shape=[12, 12])
    assert first["initial"]["pool"] == array_identity(pool)
    assert first["initial"]["cupy"] == array_identity(pool[:, :3])
    altered = start.copy()
    altered[0, 0] += 1
    with pytest.raises(ValueError, match="prefix"):
        block_inputs.identities(a, None, a, reference(), altered, 3, pool)


@pytest.mark.parametrize("missing", ["all", "operator", "reference", "pool", "warp", "cupy"])
def test_incomplete_identity_records_rejected(missing):
    a = sp.eye(12, format="csr")
    record = block_inputs.identities(a, None, a, reference(), block_inputs.initial_block(12, 4), 4)
    if missing == "all":
        record = None
    elif missing == "operator":
        del record["operators"]["operator"]
    elif missing == "reference":
        del record["reference"]
    else:
        del record["initial"][missing]
    with pytest.raises(ValueError, match="input identities"):
        block_inputs.signature(record)


@pytest.mark.parametrize("consistent", [False, True])
def test_actual_scalar_and_block_device_coefficients_are_certified(consistent):
    a = sp.diags([-np.ones(11), np.full(12, 4.0), -np.ones(11)], [-1, 0, 1], format="csr")
    mass = (sp.eye(12) + 0.02 * a).tocsr() if consistent else None
    shifted = a + 0.01 * (sp.eye(12) if mass is None else mass)
    blocks = shifted.tobsr(blocksize=(3, 3))
    gpu_shift = ws.bsr_from_triplets(
        4,
        4,
        wp.array(np.repeat(np.arange(4), np.diff(blocks.indptr)), dtype=wp.int32),
        wp.array(blocks.indices, dtype=wp.int32),
        wp.array(blocks.data, dtype=wp.mat33d),
    )
    gpu_a, gpu_mass = matrix(a), None if mass is None else matrix(mass)
    start = block_inputs.initial_block(12, 4)
    result = block_inputs.collect(gpu_a, gpu_mass, gpu_shift, a, mass, 0.01, reference(), start, 4)
    assert result == block_inputs.identities(a, mass, shifted, reference(), start, 4)
    wrong = a.copy()
    wrong.data[0] += 0.125
    with pytest.raises(ValueError, match="device operators"):
        block_inputs.collect(gpu_a, gpu_mass, gpu_shift, wrong, mass, 0.01, reference(), start, 4)
    wrong_device = matrix(wrong)
    with pytest.raises(ValueError, match="device operators"):
        block_inputs.collect(
            wrong_device, gpu_mass, gpu_shift, a, mass, 0.01, reference(), start, 4
        )
    with pytest.raises(ValueError, match="device operators"):
        block_inputs.collect(
            gpu_a,
            gpu_mass,
            matrix(shifted + 0.125 * sp.eye(12)),
            a,
            mass,
            0.01,
            reference(),
            start,
            4,
        )
    if consistent:
        with pytest.raises(ValueError, match="device operators"):
            block_inputs.collect(
                gpu_a, matrix(2 * mass), gpu_shift, a, mass, 0.01, reference(), start, 4
            )


@pytest.mark.parametrize("case,width", [("clamped_consistent", 12), ("hodge_torus", 8)])
def test_driver_records_verified_inputs_and_pool_before_graph(case, width, tmp_path, monkeypatch):
    if not hasattr(linear, "FSAI"):
        pytest.skip("driver requires pinned upstream block-Jacobi/FSAI preconditioners")
    output = tmp_path / (case + ".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_block",
            "--case",
            case,
            "--side",
            "3",
            "--modes",
            "3",
            "--width",
            str(width),
            "--initial-pool-width",
            str(width + 4),
            "--steps",
            "150",
            "--preconditioner",
            "bj",
            "--public-api",
            "--repeats",
            "1",
            "--out",
            str(output),
        ],
    )
    compare_block.main()
    report = json.loads(output.read_text())
    identity = block_inputs.signature(report["inputs"])
    assert report["accepted"]
    assert identity["initial"]["pool"]["shape"] == [report["n"], width + 4]
    assert identity["initial"]["warp"]["shape"] == [report["n"], width]
    assert identity["initial"]["cupy"] is None
    assert not report["graph"]["node_types_including_conditional_body"].get("HOST", 0)
    assert all("DtoD" in kind for kind in report["graph"]["memcpy_kinds"])
