"""Diagnostic hooks are setup-only and preserve the actual projected matrix."""

import numpy as np
import warp as wp

from benchmarks.experimental.general_trace import capture_trace
from benchmarks.upstream.adapters import complex_matrix, output
from warpack import GeneralEigensolver


def test_trace_restores_hooks_and_copies_pre_schur_projection():
    a = np.diag(np.arange(1.0, 41)) + np.diag(np.full(39, 0.1 + 0.2j), 1)
    solver = GeneralEigensolver(complex_matrix(a), 3, ncv=16, tol=1e-9)
    launch, finalize = wp.launch, solver.finalize
    graph, (trace, h, q) = capture_trace(solver, 40)
    assert wp.launch is launch
    assert solver.finalize == finalize
    trace.fill_(float("nan"))
    wp.capture_launch(graph)
    basis = output(q)
    np.testing.assert_allclose(output(h), basis.conj() @ a @ basis.T, atol=1e-12, rtol=1e-12)
    used = int(solver.iterations.numpy()[0])
    assert np.all(np.isfinite(trace.numpy()[:used]))
    np.testing.assert_array_equal(trace.numpy()[used - 1], solver.residuals.numpy())
    assert np.all(np.isnan(trace.numpy()[used:]))
