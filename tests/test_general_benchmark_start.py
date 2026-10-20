"""The original Spectra start must be independent of a preceding solve."""

import numpy as np
import warp as wp

from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.run_benchmarks import StartedGeneral


def test_changed_captured_start_resets_prior_restart_state():
    n = 40
    solver = StartedGeneral(complex_matrix(np.diag(np.arange(1.0, n + 1))), 4, ncv=16)
    solver._enable_checkpoint(True)
    solver.start_vector = wp.ones(n, dtype=wp.vec2d)
    solver.initialize()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        solver.initialize()
    for seed in (41, 81):
        initial = np.random.default_rng(seed).normal(size=n)
        solver.start_vector.assign(np.column_stack([initial, np.zeros(n)]))
        solver.restart_index.fill_(0)
        solver.restart_residual.fill_(wp.vec2d(1.0, 1.0))
        solver.residual_restarts.fill_(99)
        solver.residual_fallbacks.fill_(88)
        solver.status.fill_(2)
        solver._checkpoint.valid.fill_(1)
        solver._checkpoint.updates.fill_(99)
        wp.capture_launch(capture.graph)
        assert solver.restart_index.numpy()[0] == -1
        assert solver.residual_restarts.numpy()[0] == 0
        assert solver.residual_fallbacks.numpy()[0] == 0
        assert solver.status.numpy()[0] == 0
        assert solver._checkpoint.valid.numpy()[0] == 0
        assert solver._checkpoint.updates.numpy()[0] == 0
        q = output(solver.q)[: solver.keep]
        np.testing.assert_allclose(q[0], initial / np.linalg.norm(initial), atol=1e-14)
        np.testing.assert_allclose(q.conj() @ q.T, np.eye(solver.keep), atol=1e-12)
