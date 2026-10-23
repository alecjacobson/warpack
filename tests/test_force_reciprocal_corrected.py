"""Standalone CPU recovery must charge its factors and check both pencils."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from benchmarks.applications import force_reciprocal_corrected as corrected


@pytest.mark.parametrize("coupled_mass", [False, True])
def test_independent_factor_ownership_and_four_recovery_solves(coupled_mass, monkeypatch):
    rng = np.random.default_rng(157)
    random = rng.normal(size=(24, 24))
    h = sp.csc_matrix(random @ random.T + 10 * np.eye(24))
    mass = np.diag(np.linspace(0.5, 1.5, 24))
    if coupled_mass:
        mass += 0.01 * (random @ random.T)
    mass = sp.csc_matrix(mass)
    diagonal = np.linspace(0.2, 2, 24)
    native_splu = corrected.sla.splu
    factors = []

    class CountedFactor:
        def __init__(self, matrix):
            self.matrix = matrix.copy()
            self.factor = native_splu(matrix)
            self.calls = []
            factors.append(self)

        def solve(self, rhs):
            self.calls.append(rhs.shape)
            return self.factor.solve(rhs)

    monkeypatch.setattr(corrected.sla, "splu", CountedFactor)
    driver = corrected.CorrectedReciprocal(h, mass, diagonal)
    q = h @ sp.diags(1 / diagonal) @ h
    assert len(factors) == 2
    np.testing.assert_allclose(factors[0].matrix.toarray(), q.toarray())
    np.testing.assert_array_equal(factors[1].matrix.toarray(), h.toarray())
    values, basis, stages = driver.solve(5, 18)
    assert factors[1].calls == [(24, 5)] * 4
    exact = la.eigvalsh(q.toarray(), mass.toarray())[:5]
    np.testing.assert_allclose(values, 1 / exact, rtol=1e-12)
    np.testing.assert_allclose(basis.T @ (mass @ basis), np.eye(5), atol=1e-12)
    assert corrected.reciprocal_residual(q, mass, basis, 1 / values) < 1e-12
    assert stages["native_basis"].shape == (24, 5)
    assert stages["native_seconds"] > 0 and stages["recovery_seconds"] > 0


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("backend", ["corrected", "implicit"])
def test_standalone_report_preserves_all_physical_gates(consistent, backend):
    report = corrected.run(side=2, modes=4, consistent=consistent, backend=backend, repeats=1)
    assert report["accepted"] and report["k"] == 4
    assert report["inputs"]["initial"]["shape"] == [report["n"]]
    assert report["factor_nnz"]["h"] > 0
    assert len(report["rows"]) == 1
    for row in [report["warmup"], *report["rows"]]:
        assert row["accepted"]
        assert row["quality"]["reciprocal_residual"] < 1e-8
        assert row["quality"]["converged"] == 4
        if backend == "corrected":
            assert row["native_quality"] is not None
            assert row["seconds"] >= sum(row["stages"].values())
    if backend == "corrected":
        assert report["factor_nnz"]["q"] > 0 and report["factor_nnz"]["mass"] is None
        assert report["setup_seconds"] >= sum(report["setup_stages"].values())
    else:
        assert report["factor_nnz"]["q"] is None and report["factor_nnz"]["mass"] > 0
