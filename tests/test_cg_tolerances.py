"""Relative residual preconditioning must not silently discard small RHSs."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CGInverse


def test_relative_only_cg_replay_across_rhs_scales():
    diagonal = np.linspace(1, 2, 32)
    inverse = CGInverse(matrix(sp.diags(diagonal, format="csr")), tol=0.01, atol=0.0, maxiter=10)
    b, x = wp.ones((2, 32), dtype=wp.float64), wp.empty((2, 32), dtype=wp.float64)
    inverse.apply(b, x)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        inverse.reset_stats()
        inverse.apply(b, x)
    for scale in [1.0, 1e-8, 1e-20, 1e8]:
        source = np.full((2, 32), scale)
        b.assign(source)
        wp.capture_launch(capture.graph)
        actual = x.numpy()
        residual = np.linalg.norm(actual * diagonal - source, axis=1)
        assert np.max(residual / np.linalg.norm(source, axis=1)) < 0.01
        iterations, maximum, unconverged, count = inverse.stats.numpy()
        assert 0 < maximum <= 10 and iterations > 0 and unconverged == 0 and count == 2


def test_default_keeps_upstream_absolute_tolerance_and_zero_rhs():
    inverse = CGInverse(matrix(sp.eye(16, format="csr")), tol=0.01)
    b = wp.full((1, 16), 1e-8, dtype=wp.float64)
    x = wp.empty_like(b)
    inverse.apply(b, x)
    np.testing.assert_array_equal(x.numpy(), 0)
    assert inverse.stats.numpy()[0] == 0
    relative = CGInverse(matrix(sp.eye(16, format="csr")), tol=0.01, atol=0.0)
    b.zero_()
    relative.apply(b, x)
    np.testing.assert_array_equal(x.numpy(), 0)
    assert relative.stats.numpy()[2] == 0


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf")])
def test_invalid_absolute_tolerance(atol):
    with pytest.raises(ValueError, match="atol"):
        CGInverse(matrix(sp.eye(4, format="csr")), atol=atol)


@pytest.mark.parametrize("refinement_steps", [0, 1])
def test_adaptive_cg_capture_opt_in_and_changed_starts(tmp_path, refinement_steps):
    import re

    from warpack import SparseOperator, SymmetricEigensolver

    a = np.diag(np.arange(1.0, 41.0))
    a += np.diag(np.full(39, -0.1), 1) + np.diag(np.full(39, -0.1), -1)
    wa = matrix(a)
    default = CGInverse(wa)
    assert default.requires_static_outer_capture  # Preserve production default.
    if not hasattr(default.state, "_cur_iter_and_condition"):
        with pytest.raises(ValueError, match="preallocated loop control"):
            CGInverse(wa, adaptive_capture=True, refinement_steps=refinement_steps)
        return
    inverse = CGInverse(
        wa,
        tol=1e-13,
        maxiter=100,
        adaptive_capture=True,
        refinement_steps=refinement_steps,
        verify_residuals=bool(refinement_steps),
    )
    rng = np.random.default_rng(41)
    starts = wp.array(rng.normal(size=(8, 40)), dtype=wp.float64)
    solver = SymmetricEigensolver(
        SparseOperator(wa), 3, ncv=8, iteration_operator=inverse, initial_vectors=starts, tol=1e-10
    )
    graph = solver.capture(100)
    expected = np.linalg.eigvalsh(a)[:3]
    for _ in range(2):
        starts.assign(rng.normal(size=starts.shape))
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        np.testing.assert_allclose(w, expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(v.T @ v, np.eye(3), atol=1e-10, rtol=0)
        assert np.linalg.norm(a @ v - v * w, axis=0).max() < 1e-8
        assert solver.converged.numpy()[0] == 3
        assert solver.status.numpy()[0] == 0
        assert 0 < solver.iterations.numpy()[0] < 100
        if refinement_steps:
            assert inverse.fresh_stats.numpy()[2] == 0
    dot = tmp_path / "adaptive-cg.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text
