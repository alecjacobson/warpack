"""Singular pencils: finite/infinite classification, gauge and interior checks."""

import numpy as np
import pytest
import scipy.linalg as la

from benchmarks.applications.conformal import (
    boundary_reference,
    dense_reference,
    independent_energy,
    make_conformal_case,
    metrics,
)


@pytest.mark.parametrize("height", [0.0, 0.25])
@pytest.mark.parametrize("pin", [0, 14, 35])
def test_original_pencil_gauge_and_independent_energy(height, pin):
    case = make_conformal_case(6, 6, height, pin)
    k, m = case.stiffness.toarray(), case.mass.toarray()
    np.testing.assert_allclose(k, k.T, atol=1e-14)
    np.testing.assert_allclose(m, m.T, atol=1e-14)
    assert np.min(la.eigvalsh(k)) > -1e-13
    assert np.min(la.eigvalsh(m)) > -1e-13
    translations = la.block_diag(np.ones((36, 1)), np.ones((36, 1)))
    np.testing.assert_allclose(k @ translations, 0, atol=1e-14)
    np.testing.assert_allclose(m @ translations, 0, atol=1e-14)
    kr, mr = case.reduced
    assert la.eigvalsh((kr - case.shift * mr).toarray()).min() > 0
    assert np.linalg.matrix_rank(mr.toarray()) == 2 * (len(case.boundary) - 1)
    w, v, infinite = dense_reference(case)
    assert infinite == 2 * (6 - 2) ** 2
    reference = boundary_reference(case)
    assert metrics(case, w, v, reference)["accepted"]
    if height == 0:
        np.testing.assert_allclose(w[:2], 0, atol=1e-13)
        xy = case.xyz[:, :2] - case.xyz[case.boundary, :2].mean(axis=0)
        exact = np.column_stack((np.r_[xy[:, 0], xy[:, 1]], np.r_[-xy[:, 1], xy[:, 0]]))
        exact = exact @ la.inv(la.cholesky(exact.T @ m @ exact))
        assert la.norm(v[:, :2] - exact @ (exact.T @ m @ v[:, :2])) < 1e-10
    else:
        assert w[0] > 1e-5
    x = np.random.default_rng(7).normal(size=(72, 3))
    e, b = independent_energy(case, x)
    np.testing.assert_allclose(e, np.sum(x * (k @ x), axis=0), rtol=1e-13)
    np.testing.assert_allclose(b, np.sum(x * (m @ x), axis=0), rtol=1e-13)


def test_interior_error_cannot_hide_in_singular_mass_norm():
    case = make_conformal_case(7, 6, 0.25)
    w, v = boundary_reference(case)
    damaged = v[:, :6].copy()
    damaged[8, 0] += 0.01
    assert not metrics(case, w, damaged, (w, v))["accepted"]
    assert metrics(case, w, damaged, (w, v))["subspace_error"] > 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [dict(side=2), dict(k=3), dict(k=100), dict(pin=-1), dict(height=float("nan")), dict(shift=0)],
)
def test_invalid_case(kwargs):
    with pytest.raises(ValueError):
        make_conformal_case(**kwargs)
