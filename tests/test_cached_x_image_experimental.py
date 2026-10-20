"""Cached operator images must follow both QR/projection passes and replay inputs."""

from functools import partial

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator

CachedXBlockLOBPCG = partial(BlockLOBPCG, reuse_operator_images=True)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("kind", ["spd", "indefinite", "repeated", "rigid", "invariant"])
def test_cached_images_match_current_original_pencil(optimized, metric, kind, tmp_path):
    n, width, k = 64, 8, 4
    rng = np.random.default_rng(759)
    q = la.qr(rng.normal(size=(n, n)))[0]
    excluded = 6 if kind == "rigid" else 2 if kind == "invariant" else 0

    def fixture(changed):
        lower = np.eye(n)
        if metric:
            lower += np.diag(np.linspace(0.05, 0.3 + 0.1 * changed, n - 1), -1)
        mass = lower @ lower.T
        modes = la.solve_triangular(lower.T, q, lower=False)
        values = np.arange(1.0, n + 1)
        if kind == "indefinite":
            values -= 4
        if kind in ("repeated", "rigid"):
            values = np.ceil(values / 2)
        if kind == "rigid":
            values[:excluded] = 0
        if changed:
            values[excluded : excluded + width] += 100
        a = mass @ (modes * values) @ modes.T @ mass
        basis = (modes[:, :excluded] * np.linspace(2.0, 3.0, excluded)).T.copy()
        return a, mass, values, modes, basis

    a, mass, values, modes, basis = fixture(False)
    op, mass_op = SparseOperator(matrix(a)), SparseOperator(matrix(mass))
    pre = SparseOperator(matrix(la.inv(a + 5 * mass)))
    constraints = None if not excluded else wp.array(basis, dtype=wp.float64)
    initial = wp.array(rng.normal(size=(width, n)), dtype=wp.float64)
    solver = CachedXBlockLOBPCG(
        op,
        k,
        width=width,
        mass=mass_op,
        preconditioner=pre,
        constraints=constraints,
        normalize_constraints=bool(excluded),
        initial_vectors=initial,
        initial_perturbation=1e-3,
        tol=1e-10,
        symmetric_products=optimized,
        fused_updates=optimized,
        paired_rotation=optimized,
    )
    graph = solver.capture(100)
    for changed in (False, True):
        a, mass, values, modes, basis = fixture(changed)
        for device, host in (
            (op.matrix, a),
            (mass_op.matrix, mass),
            (pre.matrix, la.inv(a + 5 * mass)),
        ):
            csr = sp.csr_matrix(host)
            np.testing.assert_array_equal(device.columns.numpy(), csr.indices)
            device.values.assign(csr.data)
        if excluded:
            constraints.assign(basis)
        supplied = initial.numpy().copy()
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        order = np.argsort(values[excluded:])[:k] + excluded
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        np.testing.assert_allclose(w, values[order], atol=1e-9)
        np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), atol=1e-10)
        assert la.norm(a @ v - (mass @ v) * w, axis=0).max() < 2e-8
        wanted = modes[:, order]
        assert la.norm(v - wanted @ (wanted.T @ mass @ v)) < 1e-7
        # Check all carried products directly, not just the final eigenpairs.
        s, az = solver.s.numpy().T, solver.az.numpy().T
        expected = a @ s
        error = la.norm(az - expected, axis=0) / np.maximum(1, la.norm(expected, axis=0))
        assert error.max() < 5e-12
        np.testing.assert_array_equal(initial.numpy(), supplied)
        if excluded:
            normalized = solver.constraints.numpy().T
            np.testing.assert_allclose(
                solver._image_constraints.numpy().T, a @ normalized, rtol=1e-10, atol=1e-12
            )
            np.testing.assert_array_equal(constraints.numpy(), basis)
        wp.copy(initial, solver.x)
    dot = tmp_path / "cached-image.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert "DtoH" not in text and "HtoD" not in text and "HOST" not in text
