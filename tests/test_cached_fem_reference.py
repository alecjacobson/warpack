"""Cached independent references preserve FEM nullity, boundary clusters and guards."""

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.applications import fem_cases
from benchmarks.cached_fem_reference import FEMReferenceCache


@pytest.mark.parametrize(
    "name",
    [
        "clamped_lumped",
        "clamped_consistent",
        "free",
        "disconnected",
        "heterogeneous",
        "repeated_bending",
    ],
)
def test_fresh_fixture_policy_is_preserved_on_cache_reuse(tmp_path, name):
    original = fem_cases.reference
    baseline = fem_cases.make_fem_case(name, side=1, modes=5)
    with FEMReferenceCache(tmp_path) as cache:
        first = fem_cases.make_fem_case(name, side=1, modes=5)
        second = fem_cases.make_fem_case(name, side=1, modes=5)
    assert fem_cases.reference is original
    assert [r["status"] for r in cache.records] == ["created", "verified_reuse"]
    assert first.problem.k == second.problem.k == baseline.problem.k
    assert first.problem.nullity == second.problem.nullity == baseline.problem.nullity
    for result in [first, second]:
        np.testing.assert_array_equal(result.spectrum, baseline.spectrum)
        np.testing.assert_array_equal(result.vectors, baseline.vectors)
        np.testing.assert_array_equal(result.problem.null_basis, baseline.problem.null_basis)
        assert result.spectrum[result.problem.k] - result.spectrum[result.problem.k - 1] > 1e-8
    if name == "repeated_bending":
        assert first.problem.k == 6


def test_changed_actual_sparse_matrices_and_count_invalidate(tmp_path):
    a = sp.diags(np.arange(1.0, 17.0), format="csr")
    b = sp.eye(16, format="csr")
    with FEMReferenceCache(tmp_path) as cache:
        fem_cases.reference(a, b, 4)
        fem_cases.reference(a, b, 4)
        fem_cases.reference(a * 1.1, b, 4)
        fem_cases.reference(a, b * 1.1, 4)
        changed = a + sp.diags([0.1 * np.ones(15), 0.1 * np.ones(15)], [-1, 1], format="csr")
        fem_cases.reference(changed, b, 4)
        fem_cases.reference(a, b, 5)
    assert [r["status"] for r in cache.records] == [
        "created",
        "verified_reuse",
        "created",
        "created",
        "created",
        "created",
    ]
    assert len({r["key"] for r in cache.records}) == 5


def test_context_restores_reference_on_failure_and_rejects_nesting(tmp_path):
    original = fem_cases.reference
    with pytest.raises(RuntimeError):
        with FEMReferenceCache(tmp_path):
            with FEMReferenceCache(tmp_path):
                pass
    assert fem_cases.reference is original
    with pytest.raises(ValueError):
        with FEMReferenceCache(tmp_path):
            raise ValueError("caller failure")
    assert fem_cases.reference is original


def test_arpack_branch_reuses_full_guard_capacity(tmp_path):
    n = 800
    a = sp.diags(np.arange(1.0, n + 1), format="csr")
    b = sp.eye(n, format="csr")
    with FEMReferenceCache(tmp_path) as cache:
        first = fem_cases.reference(a, b, 12)
        second = fem_cases.reference(a, b, 12)
    assert cache.records[1]["status"] == "verified_reuse"
    np.testing.assert_allclose(first[0], np.arange(1.0, 13.0), atol=1e-11)
    for actual, wanted in zip(second[:2], first[:2]):
        np.testing.assert_array_equal(actual, wanted)
    assert "ARPACK" in second[2]
