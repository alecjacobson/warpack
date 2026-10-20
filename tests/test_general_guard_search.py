"""Exploratory search guards preserve requested-pair priority and replay state."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.experimental.general_frontier_search import FrontierSearchGeneral
from benchmarks.experimental.general_guard_search import (
    GuardSearchGeneral,
    select_guard_after_requested,
)
from benchmarks.experimental.general_independent_search import IndependentSearchGeneral
from benchmarks.experimental.general_restart_rounds import RestartRoundsGeneral
from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.general_reference import GeneralReference


def test_guard_selection_waits_for_requested_residuals():
    values = wp.array(np.column_stack([[10.0, 9.0, 8.0, 7.0], np.zeros(4)]), dtype=wp.vec2d)
    order = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32)
    norms = wp.zeros((4, 1), dtype=wp.float64)
    index = wp.zeros(1, dtype=wp.int32)
    selections = wp.zeros_like(index)
    for wanted, expected in ((1e-4, 0), (1e-24, 3), (np.nan, 0)):
        index.zero_()
        selections.zero_()
        norms.assign(np.array([[wanted], [1e-24], [0.02**2], [0.04**2]]))
        wp.launch(
            select_guard_after_requested, 1, [values, order, norms, index, selections, 2, 1e-10]
        )
        assert index.numpy()[0] == expected
        assert selections.numpy()[0] == int(expected == 3)


@pytest.mark.parametrize(
    "solver_type", [GuardSearchGeneral, IndependentSearchGeneral, FrontierSearchGeneral]
)
def test_guard_search_changed_matrix_and_captured_counters(tmp_path, solver_type):
    n, k = 24, 4
    rng = np.random.default_rng(751)
    q = la.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    spectra = [rng.normal(size=n) + 1j * rng.normal(size=n) for _ in range(2)]

    def dense(values):
        return q @ (np.diag(values) + np.diag(np.full(n - 1, 0.01j), 1)) @ q.conj().T

    op = complex_matrix(dense(spectra[0]))
    solver = solver_type(op, k, ncv=n, tol=1e-11)
    solver.start_vector = wp.ones(n, dtype=wp.vec2d)
    graph = solver.capture(5, adaptive=False)
    for values in (spectra[0], spectra[1], spectra[0]):
        a = dense(values)
        op.values.assign(np.stack([a.real.ravel(), a.imag.ravel()], axis=-1))
        wp.capture_launch(graph)
        assert GeneralReference(a, k).check(
            output(solver.eigenvalues), output(solver.eigenvectors).T
        )["accepted"]
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        actions = (
            solver.guard_selections
            if isinstance(solver, GuardSearchGeneral)
            else solver.independent_probes
        )
        assert 0 < actions.numpy()[0] <= 5
    dot = tmp_path / "guard.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert "HOST" not in text and "HtoD" not in text and "DtoH" not in text


@pytest.mark.parametrize(
    "solver_type", [GuardSearchGeneral, IndependentSearchGeneral, FrontierSearchGeneral]
)
def test_guard_search_retains_completed_partial_space(solver_type):
    n, k = 96, 4
    rng = np.random.default_rng(1)
    a = rng.normal(size=(n, n))
    initial = rng.normal(size=n)
    solver = solver_type(complex_matrix(a), k, ncv=24, tol=1e-10)
    solver.start_vector = wp.array(np.column_stack([initial, np.zeros(n)]), dtype=wp.vec2d)
    graph = solver.capture(60, adaptive=False)
    reference = GeneralReference(a, k)
    for _ in range(2):
        wp.capture_launch(graph)
        assert reference.check(output(solver.eigenvalues), output(solver.eigenvectors).T)[
            "accepted"
        ]
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0


def test_frontier_keeps_requested_order_and_preserves_unfinished_guard():
    from benchmarks.experimental.general_frontier_search import choose_frontier

    values = wp.array(np.column_stack([np.arange(10.0, 5.0, -1), np.zeros(5)]), dtype=wp.vec2d)
    order = wp.array(np.arange(5, dtype=np.int32), dtype=wp.int32)
    norms = wp.zeros((5, 1), dtype=wp.float64)
    ready, probe, selections = [wp.zeros(1, dtype=wp.int32) for _ in range(3)]
    cases = [
        ([0.0, 0.0, 0.0, 0.04**2, 0.05**2], [0, 1, 3, 2, 4], 1, 0, 1),
        ([0.0, 0.0, 0.02**2, 0.04**2, 0.05**2], [0, 1, 2, 3, 4], 1, 0, 1),
        ([0.0, 0.0, 0.0, 0.0, 0.0], [0, 1, 2, 3, 4], 1, 1, 0),
        ([0.01, 0.0, 0.0, 0.04**2, 0.05**2], [0, 1, 2, 3, 4], 0, 0, 0),
    ]
    for residuals, expected, wanted_ready, random_probe, count in cases:
        order.assign(np.arange(5, dtype=np.int32))
        selections.zero_()
        norms.assign(np.array(residuals)[:, None])
        wp.launch(choose_frontier, 1, [values, order, norms, ready, probe, selections, 2, 3, 1e-10])
        np.testing.assert_array_equal(order.numpy(), expected)
        assert ready.numpy()[0] == wanted_ready
        assert probe.numpy()[0] == random_probe
        assert selections.numpy()[0] == count


@pytest.mark.parametrize("rounds", [True, False])
def test_independent_rounds_preserve_snapshot_and_reset_changed_replay(tmp_path, rounds):
    n, k = 24, 4
    rng = np.random.default_rng(991)
    a = rng.normal(size=(n, n))
    op = complex_matrix(a)
    if rounds:
        solver = RestartRoundsGeneral(op, k, ncv=n, tol=1e-10, round_length=2)
    else:
        from benchmarks.experimental.general_subspace_iteration import SubspaceIterationGeneral

        solver = SubspaceIterationGeneral(op, k, ncv=n, tol=1e-10)
    solver.start_vector = wp.ones(n, dtype=wp.vec2d)
    graph = solver.capture(5, adaptive=False)
    for scale in (1.0, 2.0, 1.0):
        current = scale * a
        op.values.assign(np.column_stack([current.ravel(), np.zeros(n * n)]))
        wp.capture_launch(graph)
        assert GeneralReference(current, k).check(
            output(solver.eigenvalues), output(solver.eigenvectors).T
        )["accepted"]
        assert solver.iterations.numpy()[0] == 5
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
    dot = tmp_path / "rounds.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert "HOST" not in text and "HtoD" not in text and "DtoH" not in text
    with pytest.raises(ValueError, match="fixed budget"):
        solver.capture(5, adaptive=True)
