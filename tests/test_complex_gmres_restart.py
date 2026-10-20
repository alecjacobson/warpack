"""Explicit realified restart width, forwarding and unchanged default policy."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import complex_matrix, output
from warpack import ComplexShiftInverse, ShiftInvertEigensolver


@pytest.mark.parametrize("restart", [None, 12, 1000])
def test_complex_restart_inverse_replay_and_clamp(restart, tmp_path):
    n = 40
    a = np.diag(np.arange(1.0, n + 1)) + np.diag(np.full(n - 1, 0.1j), 1)
    op = complex_matrix(a)
    inverse = ComplexShiftInverse(
        op, backend="gmres", inner_restart=restart, inner_tol=1e-13, inner_maxiter=400
    )
    assert inverse.inner_restart == min(2 * n, 64 if restart is None else restart)
    rhs = wp.ones((2, n), dtype=wp.vec2d)
    result = wp.empty_like(rhs)
    inverse.apply_complex(rhs, result)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        inverse.apply_complex(rhs, result)
    for scale in (1.0, 1.2):
        op.values.assign(np.stack([scale * a[a != 0].real, scale * a[a != 0].imag], axis=-1))
        wp.capture_launch(capture.graph)
        x, b = output(result).T, output(rhs).T
        assert np.max(np.linalg.norm(scale * a @ x - b, axis=0) / np.linalg.norm(b, axis=0)) < 1e-10
    dot = tmp_path / "restart.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    text = dot.read_text()
    assert "HOST" not in text and "HtoD" not in text and "DtoH" not in text
    inverse.close()


def test_standard_shift_forwards_explicit_restart():
    solver = ShiftInvertEigensolver(
        complex_matrix(np.diag(np.arange(1.0, 17.0))),
        3,
        sigma=0.2j,
        ncv=16,
        backend="gmres",
        inner_restart=32,
    )
    try:
        assert solver.inverse.inner_restart == 32
        graph = solver.capture(3)
        wp.capture_launch(graph)
        assert solver.converged.numpy()[0] == 3
        np.testing.assert_allclose(np.sort(output(solver.eigenvalues).real), [1, 2, 3], atol=1e-10)
    finally:
        solver.close()


@pytest.mark.parametrize("restart", [0, -1, 1.5, True])
def test_invalid_restart_rejected(restart):
    with pytest.raises((ValueError, TypeError)):
        ComplexShiftInverse(complex_matrix(np.eye(4)), backend="gmres", inner_restart=restart)


def test_cudss_rejects_unused_restart_option():
    with pytest.raises(ValueError, match="only"):
        ComplexShiftInverse(complex_matrix(np.eye(4)), inner_restart=4)
