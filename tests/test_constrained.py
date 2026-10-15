"""Independent finite-spectrum references for non-invariant equality constraints."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import ConstrainedKrylovSchur, CuDSSInverse, SparseOperator, saddle_point_matrix


def case(consistent=False, repeated=False):
    rng = np.random.default_rng(73)
    n, r = 32, 5
    q = la.qr(rng.normal(size=(n, n)))[0]
    spectrum = np.arange(n, dtype=float)
    if repeated:
        spectrum[:6] = [1, 1, 1, 2, 2, 2]
    h = (q * spectrum) @ q.T
    t = np.diag(np.linspace(0.7, 1.9, n))
    if consistent:
        t += np.diag(np.full(n - 1, 0.2), 1)
    m = t.T @ t
    if repeated:
        m = np.eye(n)
        c = q[:, -r:].T
    else:
        c = rng.normal(size=(r, n)) * np.geomspace(0.3, 3, r)[:, None]
    z = la.null_space(c)
    values, v = la.eigh(z.T @ h @ z, z.T @ m @ z)
    return h, m, c, values, z @ v


@pytest.mark.parametrize(
    "consistent,shift,repeated",
    [(False, -0.5, False), (True, -0.5, False), (True, 8.1, False), (False, 0.0, True)],
)
def test_constrained_finite_modes_graph_replay(consistent, shift, repeated):
    h, m, c, reference, basis = case(consistent, repeated)
    k = 6
    select = np.argsort(abs(reference - shift))[:k]
    hd, md, cd = matrix(h), matrix(m), matrix(c)
    kkt = saddle_point_matrix(hd, md, cd, shift)
    inv = CuDSSInverse(kkt, 1, mtype="general", matching=True, refinement_steps=1)
    start = wp.array(np.random.default_rng(14).normal(size=len(h)), dtype=wp.float64)
    solver = ConstrainedKrylovSchur(
        SparseOperator(hd),
        SparseOperator(md),
        cd,
        inv,
        k,
        ncv=24,
        target=shift,
        tol=1e-9,
        initial_vector=start,
    )
    graph = solver.capture(12)
    try:
        for seed in (15, 16):
            wp.copy(
                start, wp.array(np.random.default_rng(seed).normal(size=len(h)), dtype=wp.float64)
            )
            wp.capture_launch(graph)
            w, v, mu = (
                solver.eigenvalues.numpy(),
                solver.eigenvectors.numpy().T,
                solver.multipliers.numpy().T,
            )
            np.testing.assert_allclose(np.sort(w), np.sort(reference[select]), atol=2e-9, rtol=0)
            np.testing.assert_allclose(v.T @ m @ v, np.eye(k), atol=2e-10)
            np.testing.assert_allclose(la.svdvals(basis[:, select].T @ m @ v), 1, atol=2e-9)
            np.testing.assert_allclose(c @ v, 0, atol=2e-9)
            residual = h @ v + c.T @ mu - (m @ v) * w
            denom = np.maximum(
                1,
                la.norm(h @ v, axis=0)
                + la.norm(c.T @ mu, axis=0)
                + abs(w) * la.norm(m @ v, axis=0),
            )
            np.testing.assert_allclose(
                solver.residuals.numpy(), la.norm(residual, axis=0) / denom, atol=2e-15
            )
            assert solver.converged.numpy()[0] == k
            assert np.max(solver.constraint_residuals.numpy()) < 1e-10
            if not repeated:
                # These modes are NOT eigenvectors of the unconstrained pencil.
                assert la.norm(h @ v - (m @ v) * w) > 0.1
    finally:
        solver.graph = graph = None
        inv.close()


@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_saddle_assembly_and_captured_changed_rhs(cooperative, width):
    h, m, c, _, _ = case(True)
    shift = 2.0
    expected = np.block([[h - shift * m, c.T], [c, np.zeros((len(c), len(c)))]])
    a = saddle_point_matrix(matrix(h), matrix(m), matrix(c), shift)
    eye = wp.array(np.eye(len(expected)), dtype=wp.float64)
    out = wp.empty_like(eye)
    SparseOperator(a).apply(eye, out)
    np.testing.assert_allclose(out.numpy().T, expected, atol=1e-14)
    inverse = CuDSSInverse(
        a,
        width,
        mtype="general",
        matching=True,
        refinement_steps=1,
        cooperative_refinement=cooperative,
    )
    rhs = wp.zeros((width, len(expected)), dtype=wp.float64)
    result = wp.empty_like(rhs)
    inverse.apply(rhs, result)
    with wp.ScopedCapture() as capture:
        inverse.apply(rhs, result)
    try:
        for seed in (2, 3):
            b = np.random.default_rng(seed).normal(size=rhs.shape)
            wp.copy(rhs, wp.array(b, dtype=wp.float64))
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(result.numpy().T, la.solve(expected, b.T), atol=1e-11)
    finally:
        capture.graph = None
        inverse.close()


def test_constrained_shape_contracts():
    h = matrix(sp.eye(12))
    c = matrix(np.ones((2, 12)))
    op = SparseOperator(h)
    with pytest.raises(ValueError, match="matching shape"):
        ConstrainedKrylovSchur(op, op, c, op, 3)
    with pytest.raises(ValueError, match="finite modes"):
        ConstrainedKrylovSchur(op, op, c, op, 10)
    with pytest.raises(ValueError, match="finite"):
        saddle_point_matrix(h, h, c, np.nan)


def test_skinning_appendix_matrices_and_modes():
    import warp.sparse as ws

    from benchmarks.applications.skinning_cases import make_skinning_case, reference, validate
    from examples.problems.skinning import (
        skinning_constraint_rows,
        skinning_stiffness,
    )
    from warpack import independent_constraints

    case = make_skinning_case(3)
    block = case.elastic_hessian.tobsr(blocksize=(3, 3))
    row = np.repeat(np.arange(block.shape[0] // 3), np.diff(block.indptr))
    hd = ws.bsr_from_triplets(
        case.n,
        case.n,
        wp.array(row, dtype=wp.int32),
        wp.array(block.indices, dtype=wp.int32),
        wp.array(block.data, dtype=wp.mat33d),
    )
    scalar = skinning_stiffness(hd)
    assert scalar.nnz_sync() == hd.nnz_sync()
    identity = wp.array(np.eye(case.n), dtype=wp.float64)
    output = wp.empty_like(identity)
    SparseOperator(scalar).apply(identity, output)
    np.testing.assert_allclose(output.numpy().T, case.h.toarray(), atol=2e-13)
    raw = skinning_constraint_rows(
        wp.array(case.vertices, dtype=wp.vec3d), wp.array(case.effective_jacobian, dtype=wp.float64)
    )
    np.testing.assert_allclose(raw.numpy(), case.raw_constraints, atol=2e-15)
    cd = independent_constraints(raw)
    assert cd.shape[0] == case.c.shape[0]
    reduced = wp.empty((case.n, cd.shape[0]), dtype=wp.float64)
    from warpack import kernels as kernels

    wp.launch(
        kernels.csr_mm,
        reduced.shape,
        [cd.offsets, cd.columns, cd.values, identity, reduced, 0.0, 1.0],
    )
    c_host = reduced.numpy().T
    np.testing.assert_allclose(c_host @ c_host.T, np.eye(cd.shape[0]), atol=2e-13)
    np.testing.assert_allclose(
        case.raw_constraints @ (np.eye(case.n) - c_host.T @ c_host), 0, atol=2e-13
    )
    md = matrix(case.m)
    inverse = CuDSSInverse(
        saddle_point_matrix(scalar, md, cd, case.shift),
        1,
        mtype="general",
        matching=True,
        refinement_steps=1,
    )
    solver = ConstrainedKrylovSchur(
        SparseOperator(scalar),
        SparseOperator(md),
        cd,
        inverse,
        6,
        ncv=32,
        target=case.shift,
        tol=1e-9,
    )
    values, vectors, _ = reference(case, 6)
    graph = solver.capture(8)
    try:
        wp.capture_launch(graph)
        result = validate(
            case, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, values, vectors
        )
        assert result["accepted"], result
        assert solver.converged.numpy()[0] == 6
    finally:
        solver.graph = graph = None
        inverse.close()


def test_constraint_rank_reduction_handles_redundancy_and_scaling():
    from warpack import independent_constraints

    rng = np.random.default_rng(43)
    rows = rng.normal(size=(5, 19))
    raw = np.vstack([rows, 1e-8 * rows, 1e6 * (rows[0] + rows[1]), np.zeros(19)])
    reduced = independent_constraints(wp.array(raw, dtype=wp.float64))
    assert reduced.shape == (5, 19)
    with pytest.raises(ValueError, match="rank"):
        independent_constraints(wp.zeros((3, 19), dtype=wp.float64))
    with pytest.raises(ValueError, match="rank"):
        independent_constraints(wp.array(np.eye(5), dtype=wp.float64))
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        independent_constraints(wp.array(raw, dtype=wp.float64))


def test_zero_mode_and_pure_warp_adaptive_replay_reject_infeasible_outputs(tmp_path):
    # Zero physical mode remains admissible; only the last two directions are constrained.
    n = 20
    h, m = np.diag(np.arange(n, dtype=float)), np.eye(n)
    c = np.eye(n)[-2:]
    shift = -0.3
    # The shift may hit an excluded eigenvalue: the KKT pencil is still invertible.
    h[-2:, -2:] = shift * np.eye(2)
    kkt = np.block([[h - shift * m, c.T], [c, np.zeros((2, 2))]])
    inverse = SparseOperator(matrix(la.inv(kkt)))  # independent test-only dense inverse
    solver = ConstrainedKrylovSchur(
        SparseOperator(matrix(h)),
        SparseOperator(matrix(m)),
        matrix(c),
        inverse,
        3,
        ncv=16,
        target=shift,
        tol=1e-9,
        initial_vector=wp.array(c[0], dtype=wp.float64),
    )
    graph = solver.capture(20, adaptive=True)
    wp.capture_launch(graph)
    np.testing.assert_allclose(np.sort(solver.eigenvalues.numpy()), [0, 1, 2], atol=1e-9)
    assert solver.converged.numpy()[0] == 3
    dot = tmp_path / "constrained.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    # Avoid importing the CuPy benchmark harness into this Warp-only graph test.
    assert "HOST" not in dot.read_text()
    # Equilibrium alone passes exactly for these forbidden unconstrained eigenvectors.
    bad = np.vstack([c, np.eye(n)[3]])
    wp.copy(solver.retained, wp.array(bad, dtype=wp.float64), count=bad.size)
    solver.finalize()
    np.testing.assert_allclose(solver.residuals.numpy(), 0, atol=1e-14)
    np.testing.assert_allclose(solver.constraint_residuals.numpy(), [1, 1, 0], atol=1e-14)
    assert solver.converged.numpy()[0] == 1


@pytest.mark.parametrize("consistent,repeated", [(False, False), (True, False), (False, True)])
def test_multiplier_images_survive_restarts_without_extra_inverse_solves(consistent, repeated):
    h, m, c, _, _ = case(consistent, repeated)
    # Nonorthogonal and differently scaled constraints must work without forming
    # normal equations, assuming orthonormal rows, or changing the user's basis.
    c *= np.geomspace(1e-2, 1e2, len(c))[:, None]
    n, r, shift = len(h), len(c), -0.5
    kkt = np.block([[h - shift * m, c.T], [c, np.zeros((r, r))]])

    class CountingInverse:
        def __init__(self):
            self.operator = SparseOperator(matrix(la.inv(kkt)))
            self.n, self.device = self.operator.n, self.operator.device
            self.calls = 0

        def apply(self, x, y):
            self.calls += x.shape[0]
            self.operator.apply(x, y)

    inverse = CountingInverse()
    solver = ConstrainedKrylovSchur(
        SparseOperator(matrix(h)),
        SparseOperator(matrix(m)),
        matrix(c),
        inverse,
        6,
        ncv=24,
        target=shift,
    )
    for seed in (11, 12):
        solver.initialize(seed)
        for _ in range(4):
            solver.step()
            calls = inverse.calls
            solver.finalize()
            assert inverse.calls == calls
            # Independent dense solves on CURRENT vectors verify all rotations
            # and the restart-seed permutation, including incomplete Ritz pairs.
            for vectors, images in (
                (solver.retained.numpy(), solver.retained_multiplier_images.numpy()),
                (solver.q.numpy()[: solver.keep], solver.multiplier_images.numpy()[: solver.keep]),
            ):
                rhs = np.vstack([m @ vectors.T, np.zeros((r, solver.keep))])
                expected = la.solve(kkt, rhs)[n:].T
                np.testing.assert_allclose(images, expected, atol=2e-10, rtol=2e-9)
            values = solver.eigenvalues.numpy()
            rhs = np.vstack([m @ solver.eigenvectors.numpy().T, np.zeros((r, solver.k))])
            expected_mu = la.solve(kkt, rhs)[n:] * (values - shift)
            np.testing.assert_allclose(
                solver.multipliers.numpy().T, expected_mu, atol=2e-10, rtol=2e-9
            )
