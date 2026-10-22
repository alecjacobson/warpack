"""Independent SPD identities, changed replay and rejection for the #134 prototype."""

import numpy as np
import pytest
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import audit
from benchmarks.experimental.coarse_correction import CoarseCG, CoarseCorrection, FixedFSAI
from benchmarks.upstream.adapters import matrix
from warpack import BatchedSparseOperator


def dense_factor(factor):
    g = factor.G
    offsets, columns, values = g.offsets.numpy(), g.columns.numpy(), g.values.numpy()
    out = np.zeros(g.shape)
    for row in range(g.nrow):
        sl = slice(offsets[row], offsets[row + 1])
        out[row, columns[sl]] = values[sl]
    return out


@pytest.mark.parametrize("balanced", [False, True])
def test_exact_spd_formula_and_changed_captured_replay(balanced, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("upstream FSAI required")
    rng = np.random.default_rng(134)
    n, rank = 17, 4
    h = rng.normal(size=(n, n))
    h = h @ h.T + np.eye(n)
    z = rng.normal(size=(rank, n))
    a = matrix(h)
    basis = wp.array(z, dtype=wp.float64)
    factor = linear.FSAI(a, max_row_size=8, reuse_pattern=True)
    smoother = FixedFSAI(factor, n)
    correction = CoarseCorrection(BatchedSparseOperator(a), basis, smoother, n, balanced=balanced)
    x = wp.array(np.eye(n), dtype=wp.float64)
    y = wp.empty_like(x)
    correction.update()
    correction.apply(x, y)
    with wp.ScopedCapture() as capture:
        correction.update()
        correction.apply(x, y)
    for scale, permutation in ((1.0, False), (1.4, True), (0.7, False)):
        changed = h * scale
        current_z = z[:, ::-1].copy() if permutation else z
        a.values.assign(changed.ravel())
        factor.update()
        basis.assign(current_z)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(correction.status.numpy(), 0)
        g = dense_factor(factor)
        s = g.T @ g
        q = current_z.T @ np.linalg.solve(current_z @ changed @ current_z.T, current_z)
        expected = s + q
        if balanced:
            expected = q + (np.eye(n) - q @ changed) @ s @ (np.eye(n) - changed @ q)
        got = y.numpy().T
        np.testing.assert_allclose(got, expected, atol=2e-13, rtol=3e-12)
        np.testing.assert_allclose(got, got.T, atol=2e-13, rtol=3e-12)
        assert np.linalg.eigvalsh(got).min() > 0
    audit(capture.graph, tmp_path / "correction.dot")


@pytest.mark.parametrize("balanced", [False, True])
def test_aliasing_rank_failure_and_recovery(balanced, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("upstream FSAI required")
    rng = np.random.default_rng(7)
    n, width, rank = 23, 5, 3
    a = matrix(np.diag(np.linspace(1.0, 4.0, n)))
    z = rng.normal(size=(rank, n))
    basis = wp.array(z, dtype=wp.float64)
    factor = linear.FSAI(a, max_row_size=4)
    correction = CoarseCorrection(
        BatchedSparseOperator(a), basis, FixedFSAI(factor, width), width, balanced=balanced
    )
    values = rng.normal(size=(width, n))
    x = wp.array(values, dtype=wp.float64)
    correction.update()
    correction.apply(x, x)
    with wp.ScopedCapture() as capture:
        correction.update()
        correction.apply(x, x)
    accepted = None
    for fail in (False, True, False):
        changed = z.copy()
        if fail:
            changed[-1] = changed[0]
        basis.assign(changed)
        x.assign(values)
        wp.capture_launch(capture.graph)
        if fail:
            assert correction.status.numpy()[0] != 0
            assert np.isnan(x.numpy()).all()
        else:
            assert correction.status.numpy()[0] == 0
            if accepted is None:
                accepted = x.numpy()
            else:
                np.testing.assert_allclose(x.numpy(), accepted, atol=1e-13, rtol=1e-13)
    audit(capture.graph, tmp_path / "recovery.dot")


@pytest.mark.parametrize("balanced", [False, True])
def test_upstream_cg_with_coarse_correction_changed_replay(balanced, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("upstream FSAI required")
    rng = np.random.default_rng(1349)
    n, width, rank = 31, 5, 4
    h = rng.normal(size=(n, n))
    h = h @ h.T + 3 * np.eye(n)
    a = matrix(h)
    basis = wp.array(rng.normal(size=(rank, n)), dtype=wp.float64)
    factor = linear.FSAI(a, max_row_size=8, reuse_pattern=True)
    correction = CoarseCorrection(
        BatchedSparseOperator(a), basis, FixedFSAI(factor, width), width, balanced=balanced
    )
    solver = CoarseCG(a, correction, width, maxiter=100, tol=1e-11)
    rhs = rng.normal(size=(width, n))
    rhs[-1] = 0  # independent RHS convergence, including a zero system
    x = wp.array(rhs, dtype=wp.float64)
    correction.update()
    solver.apply(x, x)
    with wp.ScopedCapture() as capture:
        solver.reset_stats()
        correction.update()
        solver.apply(x, x)
    for scale, rhs_scale in ((1.0, 1.0), (0.7, 1e-12), (1.4, 1.0)):
        a.values.assign((h * scale).ravel())
        factor.update()
        x.assign(rhs * rhs_scale)
        wp.capture_launch(capture.graph)
        assert correction.status.numpy()[0] == 0
        assert solver.stats.numpy()[2] == 0
        assert solver.stats.numpy()[3] == width
        np.testing.assert_allclose(
            x.numpy() / rhs_scale, np.linalg.solve(h * scale, rhs.T).T, atol=1e-10, rtol=1e-9
        )
    audit(capture.graph, tmp_path / "inner-cg.dot")


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_coarse_inner_cg_refines_generalized_fem(consistent, balanced, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("upstream FSAI required")
    from benchmarks.applications.fem_hierarchy import BeamTransfer, attach_reference, beam_problem
    from benchmarks.applications.fem_hierarchy_tuning import source_usable
    from benchmarks.applications.fem_shapes import ShapeReplay

    k, width = 6, 8
    cp, fp = beam_problem(2, width, consistent), beam_problem(4, k, consistent)
    for p in (cp, fp):
        attach_reference(p)
    coarse = ShapeReplay([(0, cp, 0)], 12, refit=False, tol=1e-3)
    coarse.solve()
    basis = wp.empty((width, fp.n), dtype=wp.float64)
    transfer = BeamTransfer(2, 4, coarse.solver.eigenvectors, basis)
    correction = None

    def factory(shifted, factor, count):
        nonlocal correction
        correction = CoarseCorrection(
            BatchedSparseOperator(shifted),
            basis,
            FixedFSAI(factor, count),
            count,
            balanced=balanced,
        )
        return CoarseCG(shifted, correction, count)

    fine = ShapeReplay([(0, fp, 0)], width, refit=False, steps=None, preconditioner_factory=factory)

    def initialize():
        transfer.apply()
        correction.update()

    graph = fine.solver.capture(200, initializer=initialize)
    for _ in range(2):
        coarse.solve()
        assert source_usable(coarse.quality(cp), width, 1e-3)
        wp.capture_launch(graph)
        assert correction.status.numpy()[0] == 0
        q = fine.quality(fp)
        assert q["accepted"], q
        assert q["residual"] < 1e-11
    audit(graph, tmp_path / "fem-inner-cg.dot")
