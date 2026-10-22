"""Independent weak-mode, weighted-metric and rank-policy reference checks."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from benchmarks.applications.weighted_snapshot_reference import weighted_reference


@pytest.mark.parametrize("coupled", [False, True])
def test_known_weak_singular_mode_without_gram_condition_squaring(coupled):
    rng = np.random.default_rng(135)
    mass = np.diag(np.geomspace(0.2, 5, 64))
    if coupled:
        d = rng.normal(size=(64, 12))
        mass += 0.1 * d @ d.T
    root = la.cholesky(mass, lower=True)
    q = np.linalg.qr(rng.normal(size=(64, 12)))[0]
    v = np.linalg.qr(rng.normal(size=(12, 12)))[0]
    singular = np.r_[np.linspace(2, 0.3, 11), 1e-8]
    basis = la.solve_triangular(root.T, q, lower=False)
    g = (basis * singular) @ v.T
    ref = weighted_reference(g, mass, 12)
    np.testing.assert_allclose(ref["singular_values"], singular, rtol=1e-7, atol=2e-16)
    assert ref["rank"] == 12
    np.testing.assert_allclose(ref["basis"].T @ mass @ ref["basis"], np.eye(12), atol=3e-14)
    assert (
        np.linalg.norm(
            ref["basis"][:, -1] - basis[:, -1] * np.sign(ref["basis"][:, -1] @ mass @ basis[:, -1])
        )
        < 2e-7
    )
    np.testing.assert_allclose(ref["energy"], np.sum(singular**2), rtol=3e-15)
    # The prior documented covariance cutoff intentionally rejects this mode.
    gram_values = la.eigvalsh(g.T @ mass @ g)
    assert np.count_nonzero(gram_values > gram_values[-1] * 1e-12) == 11


@pytest.mark.parametrize("rank", [0, 2, 5])
def test_dependent_snapshots_rank_and_expected_energy(rank):
    rng = np.random.default_rng(rank)
    g = rng.normal(size=(30, rank)) @ rng.normal(size=(rank, 8))
    m = sp.diags(np.linspace(0.5, 3, 30))
    ref = weighted_reference(g, m, 4)
    assert ref["rank"] == rank
    assert (ref["basis"] is not None) == (rank >= 4)
    np.testing.assert_allclose(ref["energy"], np.sum(g * (m @ g)), atol=1e-12)
    if ref["basis"] is not None:
        captured = np.sum((ref["basis"].T @ (m @ g)) ** 2)
        np.testing.assert_allclose(captured, sum(ref["values"]), rtol=1e-13)


def test_reference_size_guard_and_invalid_mass():
    g = np.ones((6, 2))
    # Diagonal mass never requires a dense mass factor, regardless of the cap.
    weighted_reference(g, sp.eye(6), 1, dense_limit=2)
    coupled = np.eye(6) + 0.1 * np.ones((6, 6))
    with pytest.raises(ValueError, match="size limit"):
        weighted_reference(g, coupled, 1, dense_limit=2)
    for bad in (np.zeros((6, 6)), -np.eye(6)):
        with pytest.raises(ValueError, match="positive definite"):
            weighted_reference(g, bad, 1)
    coupled[0, 1] = 0.7
    with pytest.raises(ValueError, match="symmetric"):
        weighted_reference(g, coupled, 1)
    with pytest.raises(ValueError, match="symmetric"):
        weighted_reference(g, coupled * 1e-20, 1)
