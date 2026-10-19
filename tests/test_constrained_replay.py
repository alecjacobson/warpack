"""Current constraint values must reach transpose products and physical diagnostics."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
import warp.sparse as ws
from test_duplicate_sparse import duplicate_matrix

from benchmarks.upstream.adapters import matrix
from warpack import ConstrainedKrylovSchur, SparseOperator
from warpack.constrained import _constraint_norm_parts, _constraint_norms


def stored(a, pattern=None):
    if pattern is None:
        pattern = np.ones(a.shape, dtype=bool)
    rows, cols = np.where(pattern)
    return ws.bsr_from_triplets(
        *a.shape,
        wp.array(rows, dtype=wp.int32),
        wp.array(cols, dtype=wp.int32),
        wp.array(a[rows, cols], dtype=wp.float64),
        prune_numerical_zeros=False,
    )


@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("shift", [-0.3, 2.4])
def test_constrained_changed_values_complete_replay(metric, repeated, shift, duplicates, tmp_path):
    n, r, k = 32, 5, 6
    rng = np.random.default_rng(742)
    lower = np.eye(n)
    if metric:
        lower += np.diag(np.linspace(0.05, 0.3, n - 1), -1)
    mass = lower @ lower.T
    q = la.qr(rng.normal(size=(n, n)))[0]
    modes = la.solve_triangular(lower.T, q, lower=False)
    values = np.repeat(np.arange(1.0, 12.0), 3)[:n] if repeated else np.arange(1.0, n + 1)
    h = mass @ (modes * values) @ modes.T @ mass
    if repeated:
        c = modes[:, -r:].T @ mass
        changed = (np.eye(r) + 0.1 * rng.normal(size=(r, r))) @ c
        pattern = np.ones_like(c, dtype=bool)
    else:
        pattern = np.zeros((r, n), dtype=bool)
        for i, count in enumerate([32, 25, 18, 11, 5]):
            pattern[i, rng.choice(n, count, replace=False)] = True
        c = rng.normal(size=(r, n)) * pattern
        for i in range(r):
            c[i, np.flatnonzero(pattern[i])[0]] = 0  # Stored zero later becomes nonzero.
        changed = rng.normal(size=(r, n)) * pattern
    cd = duplicate_matrix(c) if duplicates else stored(c, pattern)

    def inverse_values(current):
        return la.inv(np.block([[h - shift * mass, current.T], [current, np.zeros((r, r))]]))

    # An exact dense test-fixture inverse permits changed values with stable
    # storage. Reference eigenvectors never enter the iteration or its starts.
    inverse = SparseOperator(stored(inverse_values(c)))
    solver = ConstrainedKrylovSchur(
        SparseOperator(matrix(h)),
        SparseOperator(matrix(mass)),
        cd,
        inverse,
        k,
        ncv=n,
        target=shift,
        tol=1e-10,
    )
    graph = solver.capture(2)
    for current in [c, 2 * c, changed, c]:
        coeff = duplicate_matrix(current).values.numpy() if duplicates else current[pattern].copy()
        cd.values.assign(coeff)
        inverse.matrix.values.assign(inverse_values(current).ravel())
        wp.capture_launch(graph)
        w, v, mu = (
            solver.eigenvalues.numpy(),
            solver.eigenvectors.numpy().T,
            solver.multipliers.numpy().T,
        )
        null = la.null_space(current)
        reference, basis = la.eigh(null.T @ h @ null, null.T @ mass @ null)
        selected = np.argsort(abs(reference - shift))[:k]
        wanted = null @ basis[:, selected]
        np.testing.assert_allclose(np.sort(w), np.sort(reference[selected]), atol=2e-9, rtol=0)
        np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), atol=2e-10)
        np.testing.assert_allclose(la.svdvals(wanted.T @ mass @ v), 1, atol=2e-9)
        hv, mv, force = h @ v, mass @ v, current.T @ mu
        residual = la.norm(hv + force - mv * w, axis=0) / np.maximum(
            1, la.norm(hv, axis=0) + la.norm(force, axis=0) + abs(w) * la.norm(mv, axis=0)
        )
        feasibility = np.max(
            abs(current @ v) / (la.norm(current, axis=1)[:, None] * la.norm(v, axis=0)), axis=0
        )
        assert residual.max() < 1e-10 and feasibility.max() < 1e-10
        np.testing.assert_allclose(solver.residuals.numpy(), residual, atol=2e-14)
        np.testing.assert_allclose(solver.constraint_residuals.numpy(), feasibility, atol=2e-14)
        np.testing.assert_allclose(
            solver.constraint_norms.numpy(), la.norm(current, axis=1), rtol=3e-14
        )
        np.testing.assert_array_equal(cd.values.numpy(), coeff)
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        if not repeated:
            assert la.norm(force) > 0.1
    dot = tmp_path / "constraint-refresh.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = dot.read_text()
    assert not any(word in audit for word in ["HOST", "HtoD", "DtoH"])


@pytest.mark.parametrize("length", [7, 2049, 40001])
def test_parallel_constraint_norms_irregular_rows(length):
    lengths = np.array([length, length // 3, 0, 5])
    offsets = np.r_[0, np.cumsum(lengths)]
    values = np.random.default_rng(491).normal(size=offsets[-1])
    device = wp.array(np.r_[values, np.full(7, np.nan)], dtype=wp.float64)
    rows = wp.array(offsets, dtype=wp.int32)
    identity = wp.array(np.arange(len(values) + 7), dtype=wp.int32)
    parts = wp.empty((4, max(1, min(16, (length + 2047) // 2048))), dtype=wp.float64)
    norms = wp.empty(4, dtype=wp.float64)

    def apply():
        wp.launch_tiled(
            _constraint_norm_parts,
            dim=parts.shape,
            inputs=[rows, identity, identity, device, parts],
            block_dim=128,
        )
        wp.launch(_constraint_norms, 4, [parts, norms])

    apply()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        apply()
    for scale in [1.0, 2.0, 0.5]:
        device.assign(np.r_[values * scale, np.full(7, np.nan)])
        wp.capture_launch(cap.graph)
        expected = [la.norm(values[offsets[i] : offsets[i + 1]] * scale) for i in range(4)]
        np.testing.assert_allclose(norms.numpy(), expected, rtol=3e-14, atol=0)
