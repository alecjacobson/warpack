"""The example retry policy must honor both the certificate and native status."""

import numpy as np
import pytest
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_shapes import ShapeReplay
from benchmarks.applications.runner import accuracy
from examples.certified_recovery import Attempt, recover
from tests.test_fem_shapes import clear_second_member, crossing_members
from tests.test_solver import sparse
from warpack import LowestModeCertificate


def attempt(driver, name):
    solver = driver.solver
    return Attempt(
        name,
        driver.graph,
        solver.eigenvalues,
        solver.eigenvectors,
        solver.status,
        solver.converged,
        owner=driver,
    )


def test_bounded_recovery_from_computed_invariant_modes():
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    states = crossing_members()
    first = ShapeReplay(states, 4)
    first.solve()
    assert first.quality(states[0][1])["accepted"]
    wp.launch(
        clear_second_member,
        first.solver.eigenvectors.shape,
        [first.solver.eigenvectors, first.n // 2],
    )
    source = wp.clone(first.solver.eigenvectors)
    wp.copy(first.starts, source)
    first.update(states[1][1])
    second = ShapeReplay(states, 4, perturbation=1e-3)
    wp.copy(second.starts, source)
    second.update(states[1][1])
    _, p, guard = states[1]
    certificate = LowestModeCertificate(
        first.a,
        wp.zeros_like(source),
        wp.zeros(p.k, dtype=wp.float64),
        mass=first.mass,
        upper_bound=0.5 * (p.expected[-1] + guard),
    )
    exhausted = recover([attempt(first, "invariant")], certificate)
    assert not exhausted["accepted"] and exhausted["reason"] == "attempts_exhausted"
    assert len(exhausted["attempts"]) == 1
    assert certificate.certified.numpy()[0] == 0
    recovered = recover([attempt(first, "invariant"), attempt(second, "perturbed")], certificate)
    assert recovered["accepted"] and recovered["reason"] == "certified"
    assert len(recovered["attempts"]) == 2
    assert recovered["attempts"][0]["flags"] == certificate.GAP_UNRESOLVED
    assert recovered["attempts"][0]["converged"] == p.k
    assert recovered["attempts"][1]["flags"] == 0
    assert certificate.certified.numpy()[0] == 1
    assert accuracy(p, certificate.values.numpy(), certificate.vectors.numpy().T)["accepted"]
    # A successful first attempt must not execute a later failing one.
    early = recover([attempt(second, "perturbed"), attempt(first, "invariant")], certificate)
    assert early["accepted"] and len(early["attempts"]) == 1


@pytest.mark.parametrize(
    "case,reason",
    [
        ("unavailable", "inertia_unavailable"),
        ("inconclusive", "inertia_inconclusive"),
        ("capacity", "count_requires_new_capacity_or_gap"),
        ("invalid_basis", "non_retryable_certificate_failure"),
        ("invalid_status", "invalid_solver_status"),
        ("invalid_count", "invalid_solver_status"),
        ("exhausted_count", "attempts_exhausted"),
    ],
)
def test_non_retryable_conditions_do_not_launch_later_attempts(case, reason):
    a = sparse(np.diag([1.0, 2.0, 3.0, 4.0]))
    rows = np.eye(4)[:2]
    if case == "invalid_basis":
        rows[1] = rows[0]
    values = wp.array([1.0, 2.0], dtype=wp.float64)
    vectors = wp.array(rows, dtype=wp.float64)
    source_values, source_vectors = wp.clone(values), wp.clone(vectors)
    status = wp.array([1 if case == "invalid_status" else 0], dtype=wp.int32)
    converged = wp.array([{"invalid_count": 3, "exhausted_count": 0}.get(case, 2)], dtype=wp.int32)
    with wp.ScopedCapture() as cap:
        wp.copy(values, source_values)
        wp.copy(vectors, source_vectors)
    candidate = Attempt("exact", cap.graph, values, vectors, status, converged)
    certificate = LowestModeCertificate(
        a,
        wp.zeros_like(vectors),
        wp.zeros_like(values),
        upper_bound={"capacity": 3.5, "inconclusive": 2.0}.get(case, 2.5),
        backend=None if case == "unavailable" else "cudss",
    )
    result = recover([candidate, candidate], certificate)
    assert not result["accepted"] and result["reason"] == reason
    expected_attempts = (
        0
        if case in ("unavailable", "inconclusive", "capacity")
        else 2
        if case == "exhausted_count"
        else 1
    )
    assert len(result["attempts"]) == expected_attempts
    with wp.ScopedCapture():
        with pytest.raises(RuntimeError, match="capture"):
            recover([candidate], certificate)


@pytest.mark.parametrize("invalid", [False, True])
def test_recovery_checks_known_nullspace_together_with_each_candidate(invalid):
    mass = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spectrum = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    basis = np.eye(6) / np.sqrt(mass)[:, None]
    nullspace = wp.array(basis[:2], dtype=wp.float64)
    candidates = []
    for name, indices in [("wrong", [4, 5]), ("correct", [2, 3])]:
        source_values = wp.array(spectrum[indices], dtype=wp.float64)
        source_rows = wp.array(basis[indices], dtype=wp.float64)
        values, vectors = wp.zeros_like(source_values), wp.zeros_like(source_rows)
        with wp.ScopedCapture() as cap:
            wp.copy(values, source_values)
            wp.copy(vectors, source_rows)
        candidates.append(
            Attempt(
                name,
                cap.graph,
                values,
                vectors,
                wp.zeros(1, dtype=wp.int32),
                wp.array([2], dtype=wp.int32),
                owner=(source_values, source_rows),
            )
        )
    c = LowestModeCertificate(
        sparse(np.diag(mass * spectrum)),
        wp.zeros((2, 6), dtype=wp.float64),
        wp.zeros(2, dtype=wp.float64),
        mass=sparse(np.diag(mass)),
        nullspace=nullspace,
        upper_bound=2.5,
    )
    if invalid:
        nullspace.assign(basis[[0, 5]])  # Independent rows, but the last is not a zero mode.
        result = recover(candidates[::-1], c)
        assert not c.flags.numpy()[0] & c.INVALID_BASIS
        assert not result["accepted"] and result["reason"] == "invalid_nullspace"
        assert len(result["attempts"]) == 1
        assert result["attempts"][0]["invalid_nullspace"] == 1
        return
    result = recover(candidates, c)
    assert result["accepted"] and len(result["attempts"]) == 2
    assert result["attempts"][0]["flags"] == c.GAP_UNRESOLVED
    np.testing.assert_array_equal(nullspace.numpy(), basis[:2])
    np.testing.assert_allclose(c.values.numpy(), [1.0, 2.0])
    assert c.orthogonality_error.numpy()[0] < 1e-12
