"""Energy/stress differentiation independently checks the assembled rest tangent."""

import numpy as np
import pytest

from benchmarks.elasticity_reference import assemble_stvk_rest
from tests.test_fem import dense
from warpack.fem import assemble_rest


def energy(f, mu, lam, model):
    j = np.linalg.det(f)
    if model == "stable_neo_hookean":
        alpha = 1 + mu / (lam + mu)
        return mu / 2 * (np.sum(f * f) - 3) + (lam + mu) / 2 * (j - alpha) ** 2
    if model == "stable_with_barrier":
        mh, lh = 4 * mu / 3, lam + 5 * mu / 6
        alpha = 1 + 0.75 * mh / lh
        ic = np.sum(f * f)
        return mh / 2 * (ic - 3) + lh / 2 * (j - alpha) ** 2 - mh / 2 * np.log(ic + 1)
    if model == "stvk":
        e = (f.T @ f - np.eye(3)) / 2
        return mu * np.sum(e * e) + lam / 2 * np.trace(e) ** 2
    return mu / 2 * (np.sum(f * f) - 3) - mu * np.log(j) + lam / 2 * np.log(j) ** 2


def stress(f, mu, lam, model):
    j = np.linalg.det(f)
    if model == "stable_neo_hookean":
        alpha = 1 + mu / (lam + mu)
        return mu * f + (lam + mu) * (j - alpha) * j * np.linalg.inv(f).T
    if model == "stable_with_barrier":
        mh, lh = 4 * mu / 3, lam + 5 * mu / 6
        alpha = 1 + 0.75 * mh / lh
        return mh * (1 - 1 / (np.sum(f * f) + 1)) * f + lh * (j - alpha) * j * np.linalg.inv(f).T
    if model == "stvk":
        e = (f.T @ f - np.eye(3)) / 2
        return f @ (2 * mu * e + lam * np.trace(e) * np.eye(3))
    return mu * f + (lam * np.log(j) - mu) * np.linalg.inv(f).T


@pytest.mark.parametrize(
    "model", ["stable_neo_hookean", "stable_with_barrier", "stvk", "log_neo_hookean"]
)
@pytest.mark.parametrize("poisson", [-0.2, 0.3, 0.49])
def test_energy_stress_and_skew_tet_hessian(model, poisson):
    rng = np.random.default_rng(83)
    x = np.array([[0.3, -0.7, 0.8], [1.1, 0.2, 0.4], [-0.1, 1.3, 0.1], [0.6, 0.1, 1.9]])
    mu, lam = 10 / (2 * (1 + poisson)), 10 * poisson / ((1 + poisson) * (1 - 2 * poisson))
    # Verify PK1 by differentiating the energy at a nontrivial deformation.
    f = np.eye(3) + 0.12 * rng.normal(size=(3, 3))
    eps = 1e-30
    de = np.array(
        [np.imag(energy(f + 1j * eps * d.reshape(3, 3), mu, lam, model)) / eps for d in np.eye(9)]
    ).reshape(3, 3)
    np.testing.assert_allclose(stress(f, mu, lam, model), de, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(stress(np.eye(3), mu, lam, model), 0, atol=1e-13)
    # Obtain affine shape functions by inverting the 4x4 interpolation system.
    gradients = np.linalg.inv(np.column_stack([np.ones(4), x]))[1:].T
    volume = abs(np.linalg.det(np.column_stack([np.ones(4), x]))) / 6

    def gradient(u):
        f = (x + u.reshape(4, 3)).T @ gradients
        return (volume * gradients @ stress(f, mu, lam, model).T).ravel()

    reference = np.column_stack([np.imag(gradient(1j * eps * d)) / eps for d in np.eye(12)])
    for tet in [np.array([[0, 1, 2, 3]]), np.array([[0, 2, 1, 3]])]:
        h, _, mass = assemble_rest(x, tet, young=10, poisson=poisson, density=3)
        stvk, smass = assemble_stvk_rest(x, tet, young=10, poisson=poisson, density=3)
        np.testing.assert_allclose(dense(h), reference, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(dense(stvk), reference, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(mass.numpy(), 3 * volume / 4, rtol=1e-14)
        np.testing.assert_allclose(smass.numpy(), mass.numpy(), rtol=1e-14)


def test_consistent_mass_matches_exact_quadrature_and_lumped_row_sums():
    from benchmarks.elasticity_reference import assemble_consistent_mass

    x = np.array([[0.2, 0.1, -0.3], [1.2, 0.3, 0.4], [0.1, 1.4, -0.2], [0.3, 0.5, 1.6]])
    tet = np.array([[0, 1, 2, 3]])
    _, _, lumped = assemble_rest(x, tet, density=7)
    actual = dense(assemble_consistent_mass(x, tet, density=7))
    # Symmetric four-point tetrahedral quadrature is exact for quadratic N_i*N_j.
    a = (5 + 3 * np.sqrt(5)) / 20
    b = (5 - np.sqrt(5)) / 20
    barycentric = np.full((4, 4), b)
    np.fill_diagonal(barycentric, a)
    volume = abs(np.linalg.det(np.column_stack([np.ones(4), x]))) / 6
    scalar = 7 * volume / 4 * barycentric.T @ barycentric
    np.testing.assert_allclose(actual, np.kron(scalar, np.eye(3)), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), np.repeat(lumped.numpy(), 3), rtol=1e-14)
    assert np.linalg.eigvalsh(actual).min() > 0
