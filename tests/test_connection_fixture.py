"""Independent element, Fourier, mass and gauge checks for complex fixtures."""

import numpy as np
import pytest
import scipy.linalg as la

from benchmarks.applications.connection import make_connection_case, triangle_energy


@pytest.mark.parametrize("kind", ["identity", "lumped", "consistent"])
@pytest.mark.parametrize("copies", [1, 2])
def test_connection_energy_spectrum_and_complex_metric(kind, copies):
    case = make_connection_case(5, 8, kind, copies)
    p = case.problem
    a = p.a.toarray()
    mass = np.eye(p.n) if p.b is None else p.b.toarray()
    np.testing.assert_allclose(a, a.conj().T, atol=1e-13)
    np.testing.assert_allclose(mass, mass.conj().T, atol=1e-13)
    assert la.eigvalsh(mass).min() > 0
    assert abs(a.imag).max() > 0.1
    if kind == "consistent":
        assert abs(mass.imag).max() > 1e-5
        assert np.count_nonzero(mass - np.diag(np.diag(mass))) > p.n
    values, vectors = la.eigh(a, mass)
    np.testing.assert_allclose(values[: p.k], p.expected, rtol=1e-11, atol=1e-10)
    basis = p.requested_basis
    np.testing.assert_allclose(basis.conj().T @ mass @ basis, np.eye(basis.shape[1]), atol=1e-13)
    assert la.norm(vectors[:, : p.k] - basis @ (basis.conj().T @ mass @ vectors[:, : p.k])) < 1e-11
    rng = np.random.default_rng(918)
    x = rng.normal(size=(p.n, 3)) + 1j * rng.normal(size=(p.n, 3))
    k_energy, m_energy = triangle_energy(case, x)
    np.testing.assert_allclose(k_energy, np.sum(x.conj() * (a @ x), axis=0).real, rtol=1e-13)
    np.testing.assert_allclose(m_energy, np.sum(x.conj() * (mass @ x), axis=0).real, rtol=1e-13)
    if copies == 2:
        np.testing.assert_allclose(values[::2], values[1::2], rtol=1e-11, atol=1e-10)


def test_gauge_and_amplitude_change_preserves_generalized_spectrum():
    old = make_connection_case(6, 10, "consistent", 2, gauge_seed=11)
    new = make_connection_case(6, 10, "consistent", 2, gauge_seed=31)
    assert np.linalg.norm(old.problem.a.data - new.problem.a.data) > 1.0
    for case in [old, new]:
        values = la.eigvalsh(case.problem.a.toarray(), case.problem.b.toarray())
        np.testing.assert_allclose(values[:10], old.problem.expected, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize(
    "side,modes,kind,copies",
    [
        (2, 1, "consistent", 1),
        (4, 0, "lumped", 1),
        (4, 16, "identity", 1),
        (4, 2, "invalid", 1),
        (4, 2, "consistent", 0),
    ],
)
def test_bad_connection_configuration(side, modes, kind, copies):
    with pytest.raises(ValueError):
        make_connection_case(side, modes, kind, copies)
