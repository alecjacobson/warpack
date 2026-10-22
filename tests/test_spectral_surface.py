"""Independent energy and geometry checks for the dominant reconstruction pencil."""

import numpy as np
import pytest
import scipy.linalg as la

from benchmarks.applications.spectral_surface import (
    cube_tets,
    dense_reference,
    reconstruction_quality,
    sample_operators,
    surface_pencil,
)


def test_affine_evaluation_gradient_and_volume():
    vertices, _ = cube_tets(4)
    points = np.random.default_rng(136).uniform(-1.19, 1.19, (35, 3))
    s, g = sample_operators(points, 4)
    coefficient = np.array([0.3, -0.7, 1.2])
    field = vertices @ coefficient + 0.2
    np.testing.assert_allclose(s @ field, points @ coefficient + 0.2, atol=1e-14)
    np.testing.assert_allclose(
        (g @ field).reshape(-1, 3), np.tile(coefficient, (len(points), 1)), atol=1e-14
    )
    p = surface_pencil(4)
    assert abs(p.lumped_mass.sum() - 2.4**3) < 1e-13
    np.testing.assert_allclose(p.laplacian @ np.ones(len(vertices)), 0, atol=3e-15)
    assert abs(field @ (p.laplacian @ field) - 2.4**3 * (coefficient @ coefficient)) < 1e-13


@pytest.mark.parametrize("radii", [(0.7, 0.7, 0.7), (0.8, 0.65, 0.5)])
def test_psd_alignment_spd_nonmass_metric_and_independent_energy(radii):
    p = surface_pencil(4, radii=radii)
    u = np.random.default_rng(4).normal(size=p.a.shape[0])
    grad = (p.gradients @ u).reshape(-1, 3)
    eta = p.parameters["anisotropy"]
    direct_a = np.mean(np.sum(grad**2, axis=1) + eta * np.sum(grad * p.normals, axis=1) ** 2)
    direct_b = (
        p.parameters["alpha"] * np.mean((p.samples @ u) ** 2)
        + p.parameters["beta"] * (u @ (p.laplacian @ u))
        + p.parameters["gamma"] * np.sum((p.laplacian @ u) ** 2 / p.lumped_mass)
    )
    np.testing.assert_allclose(u @ (p.a @ u), direct_a, rtol=2e-15)
    np.testing.assert_allclose(u @ (p.b @ u), direct_b, rtol=2e-15)
    assert la.eigvalsh(p.a.toarray()).min() > -1e-12
    assert la.eigvalsh(p.b.toarray()).min() > 1e-6
    assert (p.b - p.b.T).nnz == 0
    np.testing.assert_allclose(p.a @ np.ones(p.a.shape[0]), 0, atol=2e-14)
    # Without anchors the derivative regularizers have a constant nullspace.
    regularizer = p.b - p.parameters["alpha"] * (p.samples.T @ p.samples) / len(p.points)
    np.testing.assert_allclose(regularizer @ np.ones(p.a.shape[0]), 0, atol=2e-14)


def test_dominant_objective_and_zero_set_are_separate_checks():
    p = surface_pencil(6)
    values, vectors = dense_reference(p, 3)
    assert values[0] > values[1] > values[2]
    np.testing.assert_allclose(p.a @ vectors, (p.b @ vectors) * values, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(vectors.T @ (p.b @ vectors), np.eye(3), atol=3e-14)
    geometry = reconstruction_quality(p, vectors[:, 0])
    assert geometry["valid_zero_set"]
    assert geometry["sample_distance_rms"] < 0.03
    assert geometry["zero_set_relative_radial_rms"] < 0.08
    assert geometry["normal_alignment_mean"] > 0.9
    flipped = reconstruction_quality(p, -3.7 * vectors[:, 0])
    for key in geometry:
        if key not in ("valid_zero_set", "crossings"):
            np.testing.assert_allclose(geometry[key], flipped[key], atol=1e-14)
    assert not reconstruction_quality(p, np.ones(p.a.shape[0]))["valid_zero_set"]


@pytest.mark.parametrize(
    "parameters",
    [dict(alpha=0), dict(beta=0), dict(gamma=-1), dict(anisotropy=-1), dict(alpha=float("nan"))],
)
def test_unanchored_or_invalid_metric_rejected(parameters):
    with pytest.raises(ValueError):
        surface_pencil(4, **parameters)
