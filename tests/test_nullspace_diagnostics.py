"""Returned numerical modes are evidence, not a certified topological count."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.applications.hodge import make_hodge_case
from tests.test_solver import sparse
from warpack import NumericalNullspaceDiagnostics as Diagnostics
from warpack import SparseOperator


def flags(d):
    return int(d.flags.numpy()[0])


@pytest.mark.parametrize(
    "name", ["sphere", "disk", "annulus", "torus", "disconnected", "weak_positive"]
)
def test_independent_topology_with_positive_guards(name, tmp_path):
    c = make_hodge_case(name)
    p = c.problem
    vectors = wp.array(c.vectors[:, : p.k].T.copy(), dtype=wp.float64)
    d = Diagnostics(
        SparseOperator(sparse(p.a.toarray())),
        vectors,
        SparseOperator(sparse(p.b.toarray())),
        mass_lower_bound=0.9,  # M=T.T T, T=I or I+0.05*shift, so lambda_min(M)>=0.95^2.
        expected_nullity=p.nullity,
    )
    d.run()
    with wp.ScopedCapture() as cap:
        d.run()
    path = tmp_path / "diagnostics.dot"
    wp.capture_debug_dot_print(cap.graph, str(path))
    dot = path.read_text()
    assert "HOST" not in dot and "HtoD" not in dot and "DtoH" not in dot
    for _ in range(2):
        wp.capture_launch(cap.graph)
        assert d.numerical_count.numpy()[0] == p.nullity
        assert d.weak_positive_count.numpy()[0] == 0
        assert d.positive_guard.numpy()[0] > 1e-7
        assert flags(d) == d.COMPLETENESS_UNVERIFIED
        np.testing.assert_allclose(d.values.numpy(), c.spectrum[: p.k], atol=2e-13)
        assert np.max(d.residuals.numpy()) < 1e-12
        assert d.orthogonality_error.numpy()[0] < 1e-12


def test_weak_positive_modes_are_not_silently_topological_zeros():
    c = make_hodge_case("weak_positive", weak_scale=1e-10)
    p = c.problem
    d = Diagnostics(
        SparseOperator(sparse(p.a.toarray())),
        wp.array(c.vectors[:, : p.k].T.copy(), dtype=wp.float64),
        SparseOperator(sparse(p.b.toarray())),
        mass_lower_bound=0.9,
        expected_nullity=p.nullity,
    )
    d.run()
    assert d.numerical_count.numpy()[0] > p.nullity
    assert d.weak_positive_count.numpy()[0] > 0
    assert flags(d) & d.GAP_UNRESOLVED
    assert flags(d) & d.EXPECTED_COUNT_MISMATCH
    assert flags(d) & d.COMPLETENESS_UNVERIFIED


@pytest.mark.parametrize("known", [False, True])
def test_adversarial_missing_repeated_zero_with_exact_positive_guard(known):
    # The returned two pairs are exact, but the second zero direction is absent.
    a = np.diag([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    d = Diagnostics(
        SparseOperator(sparse(a)),
        wp.array(np.eye(6)[[0, 2]], dtype=wp.float64),
        expected_nullity=2 if known else None,
    )
    d.run()
    assert d.numerical_count.numpy()[0] == 1
    assert d.positive_guard.numpy()[0] == 1
    assert np.max(d.residuals.numpy()) == 0
    assert flags(d) & d.COMPLETENESS_UNVERIFIED
    assert bool(flags(d) & d.EXPECTED_COUNT_MISMATCH) == known
    # Known nullity two requires room for both zeros plus a guard.
    assert bool(flags(d) & d.CAPACITY_EXHAUSTED) == known


@pytest.mark.parametrize("capacity", [1, 3])
def test_exhausted_capacity_and_full_dimension_remain_explicit(capacity):
    d = Diagnostics(
        SparseOperator(sparse(np.zeros((3, 3)))), wp.array(np.eye(3)[:capacity], dtype=wp.float64)
    )
    d.run()
    assert d.numerical_count.numpy()[0] == capacity
    assert flags(d) & d.CAPACITY_EXHAUSTED
    assert flags(d) & d.GAP_UNRESOLVED
    assert flags(d) & d.COMPLETENESS_UNVERIFIED
    assert bool(flags(d) & d.AT_MAX_CAPACITY) == (capacity == 3)
    assert d.suggested_capacity.numpy()[0] == 3


def test_generalized_units_and_changed_values_in_same_capture():
    rng = np.random.default_rng(611)
    t = rng.normal(size=(7, 7))
    m = t @ t.T + np.eye(7)
    # Q.T M Q=I; K=M Q Lambda Q.T M gives known generalized eigenpairs.
    lower = la.cholesky(m, lower=True)
    q = la.solve_triangular(lower.T, np.eye(7), lower=False)
    lam = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = m @ q @ np.diag(lam) @ q.T @ m
    op, mass = SparseOperator(sparse(a)), SparseOperator(sparse(m))
    vectors = wp.array(q[:, :3].T.copy(), dtype=wp.float64)
    d = Diagnostics(op, vectors, mass, mass_lower_bound=0.9, expected_nullity=1)
    d.run()
    with wp.ScopedCapture() as cap:
        d.run()
    for scale, eigenvalue, expected_count, weak, ambiguous in [
        (1.0, 0.0, 1, 0, 0),
        (2.0, 5e-9, 1, 1, 0),
        (3.0, 5e-8, 0, 0, 1),
        (4.0, 2e-6, 0, 0, 0),
    ]:
        lam[0] = eigenvalue
        current_mass = scale * m
        current_q = q / np.sqrt(scale)
        current_a = current_mass @ current_q @ np.diag(lam) @ current_q.T @ current_mass
        replacement = sparse(current_a)
        assert replacement.nnz == op.matrix.nnz
        wp.copy(op.matrix.values, replacement.values)
        wp.copy(mass.matrix.values, sparse(current_mass).values)
        wp.copy(vectors, wp.array(current_q[:, :3].T.copy(), dtype=wp.float64))
        wp.capture_launch(cap.graph)
        assert d.numerical_count.numpy()[0] == expected_count
        assert d.weak_positive_count.numpy()[0] == weak
        assert d.ambiguous_count.numpy()[0] == ambiguous
        np.testing.assert_allclose(d.values.numpy(), lam[:3], atol=2e-13)
        assert not flags(d) & (d.INVALID_DATA | d.INVALID_BASIS | d.UNCONVERGED)
        assert bool(flags(d) & d.GAP_UNRESOLVED) == bool(weak or ambiguous)
    for bad in [np.tile(current_q[:, 0], (3, 1)), np.full((3, 7), np.nan), np.zeros((3, 7))]:
        wp.copy(vectors, wp.array(bad, dtype=wp.float64))
        wp.capture_launch(cap.graph)
        assert flags(d) & (d.INVALID_BASIS | d.INVALID_DATA)
        assert d.numerical_count.numpy()[0] == 0
        assert np.isnan(d.positive_guard.numpy()[0])
    wp.copy(vectors, wp.array(current_q[:, :3].T.copy(), dtype=wp.float64))
    wp.capture_launch(cap.graph)
    assert not flags(d) & (d.INVALID_DATA | d.INVALID_BASIS)


def test_unconverged_and_negative_spectrum_evidence():
    a = np.diag([-1.0, 0.0, 2.0, 4.0])
    rows = np.eye(4)[[0, 1, 2]]
    rows[2] = np.array([0.0, 0.0, 1.0, 1.0]) / np.sqrt(2)
    d = Diagnostics(SparseOperator(sparse(a)), wp.array(rows, dtype=wp.float64))
    d.run()
    assert flags(d) & d.NEGATIVE_SPECTRUM
    assert flags(d) & d.UNCONVERGED
    assert d.numerical_count.numpy()[0] == 1
    assert d.ambiguous_count.numpy()[0] == 2


@pytest.mark.parametrize(
    "bad",
    [
        "shape",
        "empty",
        "dtype",
        "mass",
        "missing_bound",
        "bound",
        "threshold",
        "expected",
        "identity_bound",
    ],
)
def test_diagnostic_layout_and_scaling_contracts(bad):
    op = SparseOperator(sparse(np.eye(4)))
    mass = SparseOperator(sparse(np.eye(3 if bad == "mass" else 4)))
    shape = (2, 3) if bad == "shape" else (0, 4) if bad == "empty" else (2, 4)
    vectors = wp.zeros(shape, dtype=wp.float32 if bad == "dtype" else wp.float64)
    options = dict(mass=mass, mass_lower_bound=0.9)
    if bad == "missing_bound":
        options.pop("mass_lower_bound")
    if bad == "bound":
        options["mass_lower_bound"] = float("nan")
    if bad == "threshold":
        options["positive_threshold"] = 1e-9
    if bad == "expected":
        options["expected_nullity"] = 5
    if bad == "identity_bound":
        options.update(mass=None, mass_lower_bound=2)
    with pytest.raises(ValueError):
        Diagnostics(op, vectors, **options)


def test_diagnostics_composed_with_captured_hodge_solver():
    from benchmarks.applications.runner import warp_setup
    from benchmarks.upstream.adapters import matrix

    c = make_hodge_case("disconnected")
    p = c.problem
    solver, inverse = warp_setup(p)
    d = Diagnostics(
        SparseOperator(matrix(p.a)),
        solver.eigenvectors,
        SparseOperator(matrix(p.b)),
        mass_lower_bound=0.9,
        expected_nullity=p.nullity,
    )
    graph = solver.capture(100, finalizer=d.run)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            assert d.numerical_count.numpy()[0] == 4
            assert d.positive_guard.numpy()[0] > 1e-7
            assert flags(d) == d.COMPLETENESS_UNVERIFIED
    finally:
        graph = solver.graph = None
        inverse.close()
