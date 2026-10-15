"""Metric QR must preserve weak independent directions under captured recovery."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import GeneralizedEigensolver, SparseOperator


def make_solver():
    import scipy.sparse as sp

    n = 32
    mass = np.diag(np.linspace(1, 2, n))
    mass += np.diag(np.full(n - 1, 0.05), 1) + np.diag(np.full(n - 1, 0.05), -1)
    identity = SparseOperator(matrix(sp.eye(n, format="csr")))
    solver = GeneralizedEigensolver(
        identity, SparseOperator(matrix(sp.csr_matrix(mass))), identity, 2, ncv=4
    )
    return solver, mass


@pytest.mark.parametrize("scale", [1e-60, 1.0, 1e60])
def test_captured_metric_qr_recovers_span_across_units(scale):
    s, mass = make_solver()
    weak = np.zeros((4, 32))
    weak[:, 0] = 1
    weak[1, 1], weak[2, 2], weak[3, 3] = 1e-11, 1e-10, 1e-9
    ordinary = np.eye(32)[:4]
    s.q.assign(weak * scale)
    s.orthogonalize()  # compile both paths before capture
    with wp.ScopedCapture() as cap:
        s.status.zero_()
        s.orthogonalization_recoveries.zero_()
        s.orthogonalize()
    for block, needs_recovery in [(weak, True), (ordinary, False), (weak, True)]:
        s.q.assign(block * scale)
        wp.capture_launch(cap.graph)
        q = s.q.numpy()
        assert s.status.numpy()[0] == 0
        assert (s.orthogonalization_recoveries.numpy()[0] > 0) == needs_recovery
        np.testing.assert_allclose(q @ mass @ q.T, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(q[:, 4:], 0, atol=0)


@pytest.mark.parametrize("invalid", ["zero", "nan", "inf"])
def test_invalid_metric_blocks_report_failure(invalid):
    s, _ = make_solver()
    block = np.zeros((4, 32))
    if invalid != "zero":
        block[0, 0] = float(invalid)
    s.q.assign(block)
    s.orthogonalize()
    wp.synchronize()
    assert s.status.numpy()[0] != 0


def test_recovery_can_be_nested_in_adaptive_graph():
    import scipy.sparse as sp

    # A pure Warp inverse keeps the outer convergence loop capturable.
    a = np.diag(np.arange(1.0, 33.0))
    op = SparseOperator(matrix(sp.csr_matrix(a)))
    mass = SparseOperator(matrix(sp.eye(32, format="csr")))
    inverse = SparseOperator(matrix(sp.diags(1 / np.diag(a), format="csr")))
    s = GeneralizedEigensolver(op, mass, inverse, 2, ncv=8)
    graph = s.capture(100, adaptive=True)
    wp.capture_launch(graph)
    assert s.status.numpy()[0] == 0
    assert s.converged.numpy()[0] == 2
    np.testing.assert_allclose(s.eigenvalues.numpy(), [1, 2], atol=1e-9)
