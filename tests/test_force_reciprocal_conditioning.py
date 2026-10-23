"""CPU comparison corrections retain the original generalized covariance pencil."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from benchmarks.applications.force_covariance_compare import HostCovariance
from benchmarks.applications.force_reciprocal_conditioning import covariance_ritz, diagnose


@pytest.mark.parametrize("coupled_mass", [False, True])
@pytest.mark.parametrize("inverse_step", [False, True])
def test_covariance_recovery_from_mixed_nonorthogonal_subspace(coupled_mass, inverse_step):
    rng = np.random.default_rng(153)
    n, k = 24, 6
    h = np.diag(np.geomspace(1, 100, n))
    random = rng.normal(size=(n, n))
    prior = random @ random.T + np.eye(n)
    mass = np.diag(np.linspace(0.5, 2, n))
    if coupled_mass:
        mass += 0.02 * (random @ random.T)
    inv_h = np.diag(1 / np.diag(h))
    covariance = mass @ inv_h @ prior @ inv_h @ mass
    eigenvalues, eigenvectors = la.eigh(covariance, mass)
    expected, basis = eigenvalues[-k:][::-1], eigenvectors[:, -k:][:, ::-1]
    mixing = np.eye(k) + 0.1 * rng.normal(size=(k, k))
    mixing *= np.linspace(0.2, 2.0, k)
    host = HostCovariance(sp.csr_matrix(h), sp.csr_matrix(mass), sp.csr_matrix(prior))
    values, vectors = covariance_ritz(host, basis @ mixing, inverse_step=inverse_step)
    np.testing.assert_allclose(values, expected, rtol=1e-12)
    np.testing.assert_allclose(vectors.T @ mass @ vectors, np.eye(k), atol=1e-12)
    residual = covariance @ vectors - (mass @ vectors) * values
    assert np.linalg.norm(residual) / np.linalg.norm(covariance @ vectors) < 1e-12
    assert np.linalg.norm(vectors - basis @ (basis.T @ mass @ vectors)) < 1e-11


@pytest.mark.parametrize("consistent", [False, True])
def test_conditioning_report_requires_both_physical_equations(consistent):
    report = diagnose(side=2, modes=4, consistent=consistent)
    assert report["n"] > report["k"] == 4
    assert len(report["rows"]) == (7 if report["wider_residual_checks"] else 5)
    final = report["rows"][-1]
    assert final["accepted"], final
    for row in report["rows"]:
        q = row["quality"]
        assert q["converged"] == 4
        assert row["accepted"] == (
            q["accepted"]
            and all(value < 1e-8 for key, value in q.items() if key.endswith("q_residual"))
        )
    if report["wider_residual_checks"]:
        assert report["longdouble_mantissa_bits"] > report["float64_mantissa_bits"]
        assert "wide_implicit_q_residual" in final["quality"]
