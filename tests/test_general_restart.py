"""A converged guard must not starve an unconverged requested eigenvector."""

import numpy as np
import pytest
import warp as wp

from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.general_restart import ResidualRestartGeneral
from benchmarks.upstream.adapters import complex_matrix, output


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("nonnormal", [False, True])
def test_converged_guard_recovers_unfinished_target(scale, nonnormal, tmp_path):
    n, k, m = 96, 4, 16
    rng = np.random.default_rng(281)
    unitary = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    values = np.arange(1.0, n + 1) + 0.1j * np.sin(np.arange(n))
    triangular = np.diag(values)
    if nonnormal:
        triangular += np.diag(np.full(n - 1, 0.02 + 0.01j), 1)
    a = scale * (unitary @ triangular @ unitary.conj().T)
    solver = ResidualRestartGeneral(complex_matrix(a), k, ncv=m, which="SR", tol=1e-12)
    initial = wp.zeros_like(solver.q)

    def initialize(seed=42):
        solver.status.zero_()
        solver.restart_index.fill_(-1)
        solver.recoveries.zero_()
        wp.copy(solver.q, initial)

    solver.initialize = initialize

    def set_start(angle):
        basis = np.zeros((m, n), dtype=complex)
        basis[: solver.keep] = unitary[:, : solver.keep].T
        basis[0] = (unitary[:, 0] + angle * unitary[:, -1]) / np.sqrt(1 + angle**2)
        initial.assign(np.stack([basis.real, basis.imag], axis=-1))

    set_start(1e-9)
    graph = solver.capture(20)
    for angle in (1e-9, -2e-9):
        set_start(angle)
        wp.capture_launch(graph)
        w, v = output(solver.eigenvalues), output(solver.eigenvectors).T
        assert solver.recoveries.numpy()[0] > 0
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == k, solver.residuals.numpy()
        np.testing.assert_allclose(w, scale * values[:k], rtol=2e-12, atol=2e-12)
        assert np.max(np.linalg.norm(a @ v - v * w, axis=0) / np.maximum(1, abs(w))) < 1e-12
        np.testing.assert_allclose(np.linalg.norm(v, axis=0), 1, atol=1e-12)
    dot = tmp_path / "general-recovery.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
