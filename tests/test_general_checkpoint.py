"""Public fixed-budget preservation, replay reset and original-equation gates."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.general_reference import GeneralReference
from warpack import GeneralEigensolver, ShiftInvertEigensolver


def complex_array(values):
    values = np.asarray(values)
    return wp.array(np.stack([values.real, values.imag], axis=-1), dtype=wp.vec2d)


@pytest.mark.parametrize("which", ["LM", "SM", "LR", "SR", "LI", "SI"])
def test_fixed_checkpoint_changed_requested_set(which):
    n, k = 24, 4
    rng = np.random.default_rng(837)
    q = la.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    first = rng.normal(size=n) + 1j * rng.normal(size=n)
    second = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.01

    def dense(values):
        return q @ (np.diag(values) + np.diag(np.full(n - 1, 0.005 + 0.001j), 1)) @ q.conj().T

    op = complex_matrix(dense(first))
    starts = complex_array(np.ones(n, dtype=complex))
    solver = GeneralEigensolver(op, k, ncv=n, which=which, tol=1e-11, initial_vector=starts)
    graph = solver.capture(4, adaptive=False)
    for values in (first, second, first):
        a = dense(values)
        op.values.assign(np.stack([a.real.ravel(), a.imag.ravel()], axis=-1))
        wp.capture_launch(graph)
        w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
        keys = {
            "LM": -abs(values),
            "SM": abs(values),
            "LR": -values.real,
            "SR": values.real,
            "LI": -values.imag,
            "SI": values.imag,
        }
        expected = values[np.argsort(keys[which])[:k]]
        cost = abs(w[:, None] - expected)
        i, j = linear_sum_assignment(cost)
        assert cost[i, j].max() < 1e-10
        residuals = la.norm(a @ v - v * w, axis=0) / np.maximum(1, abs(w))
        assert residuals.max() < 1e-10
        np.testing.assert_allclose(solver.residuals.numpy(), residuals, atol=1e-13)
        np.testing.assert_allclose(la.norm(v, axis=0), 1, atol=1e-12)
        assert la.svdvals(v)[-1] > 0.1
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        assert solver._checkpoint.valid.numpy()[0] == 1
        assert 1 <= solver._checkpoint.updates.numpy()[0] <= 4
        np.testing.assert_array_equal(output(solver.ritz), np.diag(output(solver.h)))


def test_fixed_checkpoint_preserves_partial_space_and_fresh_final_gate(tmp_path):
    n, k = 96, 4
    rng = np.random.default_rng(1)
    a = rng.normal(size=(n, n))
    initial = complex_array(rng.normal(size=n).astype(complex))
    op = complex_matrix(a)
    solver = GeneralEigensolver(op, k, ncv=24, tol=1e-10, initial_vector=initial)
    graph = solver.capture(60, adaptive=False)
    reference = GeneralReference(a, k)
    for _ in range(3):
        wp.capture_launch(graph)
        assert reference.check(output(solver.eigenvalues), output(solver.eigenvectors).T)[
            "accepted"
        ]
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        np.testing.assert_array_equal(output(solver.ritz), np.diag(output(solver.h)))
    # A previously accepted snapshot cannot bypass a changed original equation.
    op.values.assign(np.column_stack([3 * a.ravel(), np.zeros(n * n)]))
    solver.finalize()
    assert solver.converged.numpy()[0] == 0
    assert solver.residuals.numpy().min() > 0.5
    dot = tmp_path / "checkpoint.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = dot.read_text()
    assert "HOST" not in audit and "DtoH" not in audit and "HtoD" not in audit


def test_checkpoint_reset_and_adaptive_opt_out():
    n, k = 64, 3
    rng = np.random.default_rng(381)
    q = la.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    easy = (q[:, :k] * [5, 7, 9]) @ q[:, :k].conj().T
    hard = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    op = complex_matrix(easy)
    solver = GeneralEigensolver(
        op,
        k,
        ncv=12,
        which="LM",
        tol=1e-11,
        initial_vector=complex_array(np.ones(n, dtype=complex)),
    )
    adaptive = solver.capture(3, adaptive=True)
    assert solver._checkpoint is None
    graph = solver.capture(1, adaptive=False)
    for a, good in ((easy, True), (hard, False), (easy, True)):
        op.values.assign(np.stack([a.real.ravel(), a.imag.ravel()], axis=-1))
        wp.capture_launch(graph)
        assert bool(solver._checkpoint.valid.numpy()[0]) == good
        assert bool(solver.converged.numpy()[0] == k) == good
    # An earlier adaptive graph remains usable after lazy checkpoint allocation.
    wp.capture_launch(adaptive)
    assert solver.converged.numpy()[0] == k
    legacy = GeneralEigensolver(op, k, ncv=12, preserve_converged=False)
    legacy.capture(2, adaptive=False)
    assert legacy._checkpoint is None


def test_checkpoint_resolves_cluster_with_loose_requested_tolerance():
    values = np.array([1e6 + 3, 1e6 + 2, 1e6 + 1, 1e6 + 0.001, 1e6, 0])
    solver = GeneralEigensolver(complex_matrix(np.diag(values)), 4, ncv=6, tol=1e-10)
    solver._enable_checkpoint(True)
    solver.order.assign(np.arange(6, dtype=np.int32))
    for indices in ([0, 1, 2, 4], [0, 1, 2, 3]):
        solver.eigenvectors.assign(np.stack([np.eye(6)[indices], np.zeros((4, 6))], axis=-1))
        candidate = np.r_[values[indices], [0, 0]]
        solver.ritz.assign(np.column_stack([candidate, np.zeros(6)]))
        solver.norms.zero_()
        solver._checkpoint.consider(solver)
    assert solver._checkpoint.updates.numpy()[0] == 2
    solver.finalize()
    np.testing.assert_array_equal(output(solver.eigenvalues).real, values[:4])
    assert solver.converged.numpy()[0] == 4


def test_shift_checkpoint_requires_original_equation():
    values = np.arange(1, 9, dtype=float)
    solver = ShiftInvertEigensolver(
        complex_matrix(np.diag(values)), 2, ncv=6, backend="gmres", tol=1e-10
    )
    try:
        core = solver.core
        core._enable_checkpoint(True)
        core.order.assign(np.arange(6, dtype=np.int32))
        core.eigenvectors.assign(np.stack([np.eye(8)[:2], np.zeros((2, 8))], axis=-1))
        for candidate, accepted in (
            ([0.01, 0.02], False),
            ([1.0, 0.5], True),
            ([1000.0, 2000.0], False),
        ):
            raw = np.r_[candidate, np.zeros(4)]
            core.ritz.assign(np.column_stack([raw, np.zeros(6)]))
            # Model a perfect inverse Ritz residual even for the wrong physical pair.
            core.norms.zero_()
            previous = core._checkpoint.updates.numpy()[0]
            core._checkpoint.consider(core)
            assert core._checkpoint.updates.numpy()[0] == previous + int(accepted)
        core.finalize()
        solver.finalize()
        np.testing.assert_allclose(output(solver.eigenvalues), [1.0, 2.0], atol=1e-12)
        assert solver.converged.numpy()[0] == 2
        assert solver.status.numpy()[0] == 0
    finally:
        solver.close()


@pytest.mark.parametrize("shifted", [False, True])
def test_checkpoint_does_not_retain_solver_in_reference_cycle(shifted):
    import weakref

    op = complex_matrix(np.diag(np.arange(1.0, 9.0)))
    solver = (
        ShiftInvertEigensolver(op, 2, ncv=6, backend="gmres")
        if shifted
        else GeneralEigensolver(op, 2, ncv=6)
    )
    solver.capture(2) if shifted else solver.capture(2, adaptive=False)
    assert (solver.core if shifted else solver)._checkpoint is not None
    reference = weakref.ref(solver)
    if shifted:
        solver.close()
    del solver
    # Dropping a solver must release its workspaces without a cyclic-GC pass.
    assert reference() is None
