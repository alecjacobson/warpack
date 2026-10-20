"""Independent damped and gyroscopic fixture/acceptance checks (CPU only)."""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment

from benchmarks.applications.damped import coupled_chain, quality


@pytest.mark.parametrize("kind", ["proportional", "nonproportional", "gyroscopic", "coupled-mass"])
def test_damped_reference_matches_full_generalized_pencil(kind):
    case = coupled_chain(nodes=12, k=8, kind=kind)
    n = 2 * case.nodes
    k, m = case.stiffness.toarray(), case.mass.toarray()
    d = (case.damping + case.gyroscopic).toarray()
    pencil_a = np.block([[np.zeros((n, n)), np.eye(n)], [-k, -d]])
    pencil_b = la.block_diag(np.eye(n), m)
    full = la.eigvals(pencil_a, pencil_b)
    costs = abs(full[:, None] - case.full_spectrum)
    rows, cols = linear_sum_assignment(costs)
    assert costs[rows, cols].max() < 1e-10
    assert quality(case, case.values, case.right)["accepted"]
    left_residual = case.a.T @ case.left - case.left * case.values.conj()
    assert np.linalg.norm(left_residual) < 1e-10
    np.testing.assert_allclose(case.left.conj().T @ case.right, np.eye(case.k), atol=1e-11)
    assert np.linalg.eigvalsh(m).min() > 0
    assert np.linalg.eigvalsh(k).min() > 0
    assert np.linalg.eigvalsh(case.damping.toarray()).min() > 0
    np.testing.assert_allclose(case.gyroscopic.toarray().T, -case.gyroscopic.toarray())
    if kind != "proportional":
        c = case.damping.toarray()
        assert np.linalg.norm(k @ c - c @ k) > 1


def test_damped_quality_rejects_missing_modes_and_bad_vectors():
    case = coupled_chain()
    values, vectors = case.values.copy(), case.right.copy()
    values[-1], vectors[:, -1] = values[0], vectors[:, 0]
    assert not quality(case, values, vectors)["accepted"]
    assert not quality(case, case.values, case.right * 0.1)["accepted"]
    assert not quality(case, case.values, np.zeros_like(case.right))["accepted"]
    vectors[0, 0] = np.nan
    assert not quality(case, values, vectors)["accepted"]


def test_recorded_physical_convergence_is_not_rejected_by_tighter_core_count():
    import json
    from pathlib import Path

    from benchmarks.applications.damped_compare import accepted_physical

    recorded = json.loads(
        (Path(__file__).parent / "fixtures/damped-physical-count.json").read_text()
    )
    quality, stats, k = recorded["quality"], recorded["statistics"], recorded["k"]
    assert stats["count"] == k and stats["core_count"] == k - 1
    assert accepted_physical(quality, stats, k)
    assert not accepted_physical(quality, dict(stats, count=k - 1), k)
    assert not accepted_physical(dict(quality, accepted=False), stats, k)
    assert not accepted_physical(quality, dict(stats, status=2), k)


def test_balanced_coordinates_preserve_original_modes():
    from benchmarks.applications.damped_backends import coordinates, physical

    case = coupled_chain(nodes=12, k=8, kind="gyroscopic")
    scale = 13.0
    balanced = coordinates(case, scale)
    vectors = case.right.copy()
    vectors[: len(vectors) // 2] *= scale
    assert np.linalg.norm(balanced @ vectors - vectors * case.values) < 1e-10
    restored = physical(vectors, scale)
    assert quality(case, case.values, restored)["accepted"]


def test_velocity_recovery_still_checks_displacement_equation():
    from benchmarks.applications.damped_backends import physical

    case = coupled_chain(nodes=12, k=8, kind="nonproportional")
    vectors = case.right.copy()
    vectors[len(vectors) // 2 :] = 0
    recovered = physical(vectors, 1.0, case.values, kinematic=True)
    assert quality(case, case.values, recovered)["accepted"]
    vectors[0, 0] += 0.1
    wrong = physical(vectors, 1.0, case.values, kinematic=True)
    assert not quality(case, case.values, wrong)["accepted"]
