"""Original equations and changed replay state for converged-set preservation."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.experimental.general_checkpoint import CheckpointGeneral
from benchmarks.upstream.adapters import complex_matrix, output


@pytest.mark.parametrize("which", ["LM", "SM", "LR", "SR", "LI", "SI"])
def test_checkpoint_replay_changes_matrix_and_requested_set(which):
    n, k = 24, 4
    rng = np.random.default_rng(837)
    unitary = la.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    first = rng.normal(size=n) + 1j * rng.normal(size=n)
    second = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.01

    def matrix(values):
        triangular = np.diag(values) + np.diag(np.full(n - 1, 0.005 + 0.001j), 1)
        return unitary @ triangular @ unitary.conj().T

    a = matrix(first)
    op = complex_matrix(a)
    solver = CheckpointGeneral(op, k, ncv=n, which=which, tol=1e-11)
    solver.start_vector = wp.ones(n, dtype=wp.vec2d)
    graph = solver.capture(4, adaptive=False)
    for values in (first, second, first):
        a = matrix(values)
        op.values.assign(np.column_stack([a.real.ravel(), a.imag.ravel()]))
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
        costs = abs(w[:, None] - expected)
        rows, cols = linear_sum_assignment(costs)
        assert costs[rows, cols].max() < 1e-10
        assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1e-10
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
        assert la.svdvals(v)[-1] > 0.1
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        assert solver.checkpoint_valid.numpy()[0] == 1
        assert solver.checkpoint_updates.numpy()[0] >= 1


def test_checkpoint_cannot_reuse_previous_convergence_on_new_problem():
    n, k = 64, 3
    rng = np.random.default_rng(381)
    q = la.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    easy = (q[:, :k] * [5, 7, 9]) @ q[:, :k].conj().T
    hard = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    op = complex_matrix(easy)
    solver = CheckpointGeneral(op, k, ncv=12, which="LM", tol=1e-11)
    solver.start_vector = wp.ones(n, dtype=wp.vec2d)
    graph = solver.capture(1, adaptive=False)
    for a, should_converge in ((easy, True), (hard, False), (easy, True)):
        op.values.assign(np.column_stack([a.real.ravel(), a.imag.ravel()]))
        wp.capture_launch(graph)
        assert bool(solver.checkpoint_valid.numpy()[0]) == should_converge
        assert bool(solver.converged.numpy()[0] == k) == should_converge
        w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
        errors = la.norm(a @ v - v * w, axis=0) / np.maximum(1, abs(w))
        np.testing.assert_allclose(solver.residuals.numpy(), errors, atol=1e-13)
        if should_converge:
            np.testing.assert_allclose(np.sort(w.real), [5, 7, 9], atol=1e-10)
        else:
            assert errors.max() > 1e-4


def test_fixed_restarts_preserve_completed_partial_spectrum(tmp_path):
    # The unmodified ncv=24 / fixed-60 profile can replace a converged
    # conjugate-boundary mode with an inaccurate larger Ritz estimate. A
    # per-cycle trace of this deterministic input first passed at cycle 15,
    # then failed ten later selections, including cycle 60.
    from benchmarks.upstream.general_reference import GeneralReference

    n, k = 96, 4
    rng = np.random.default_rng(1)
    a = rng.normal(size=(n, n))
    initial = rng.normal(size=n)
    solver = CheckpointGeneral(complex_matrix(a), k, ncv=24, tol=1e-10)
    solver.start_vector = wp.array(np.column_stack([initial, np.zeros(n)]), dtype=wp.vec2d)
    graph = solver.capture(60, adaptive=False)
    reference = GeneralReference(a, k)
    for _ in range(3):
        wp.capture_launch(graph)
        assert reference.check(output(solver.eigenvalues), output(solver.eigenvectors).T)[
            "accepted"
        ]
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        # Restoring output must not corrupt the working projected spectrum.
        np.testing.assert_array_equal(output(solver.ritz), np.diag(output(solver.h)))
    dot = tmp_path / "general-checkpoint.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = dot.read_text()
    assert "HOST" not in audit and "HtoD" not in audit and "DtoH" not in audit


def test_checkpoint_recognizes_resolved_cluster_improvement():
    """Exact genuine eigenpairs must not be tied by a loose requested tolerance."""
    from benchmarks.experimental.general_checkpoint import choose_checkpoint, save_checkpoint

    values = np.array([1e6 + 3, 1e6 + 2, 1e6 + 1, 1e6 + 0.001, 1e6, 0])
    n, k = len(values), 4
    solver = CheckpointGeneral(complex_matrix(np.diag(values)), k, ncv=n, tol=1e-10)
    solver.order.assign(np.arange(n, dtype=np.int32))
    candidates = []
    for indices in ([0, 1, 2, 4], [0, 1, 2, 3]):
        vectors = np.eye(n)[indices]
        candidates.append(
            (
                wp.array(np.stack([vectors, np.zeros_like(vectors)], axis=-1), dtype=wp.vec2d),
                wp.array(
                    np.column_stack([np.r_[values[indices], np.zeros(n - k)], np.zeros(n)]),
                    dtype=wp.vec2d,
                ),
            )
        )

    def record_exact_candidates():
        solver.checkpoint_valid.zero_()
        solver.checkpoint_updates.zero_()
        solver.status.zero_()
        # Both candidates are exact eigenpairs of the same original matrix;
        # their cached residuals are exactly zero, with unit independent vectors.
        solver.norms.zero_()
        for vectors, candidate_values in candidates:
            wp.copy(solver.eigenvectors, vectors)
            wp.copy(solver.ritz, candidate_values)
            wp.launch(
                choose_checkpoint,
                1,
                [
                    solver.ritz,
                    solver.order,
                    solver.norms,
                    solver.status,
                    solver.checkpoint_valid,
                    solver.checkpoint_replace,
                    solver.checkpoint_score,
                    solver.checkpoint_error,
                    solver.checkpoint_scale,
                    solver.checkpoint_updates,
                    solver.which,
                    solver.tol,
                ],
            )
            wp.launch(
                save_checkpoint,
                solver.eigenvectors.shape,
                [
                    solver.eigenvectors,
                    solver.ritz,
                    solver.order,
                    solver.checkpoint_replace,
                    solver.checkpoint_vectors,
                    solver.checkpoint_values,
                ],
            )
        solver.finalize()

    record_exact_candidates()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        record_exact_candidates()
    wp.capture_launch(capture.graph)
    assert solver.converged.numpy()[0] == k
    np.testing.assert_array_equal(solver.residuals.numpy(), 0)
    np.testing.assert_allclose(
        np.sort(output(solver.eigenvalues).real), np.sort(values[:k]), atol=1e-8, rtol=0
    )
