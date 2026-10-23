"""Opt-in compact graphs preserve complete generalized modes and changed replay."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator
from warpack._compact_jacobi import CompactJacobi
from warpack.block import Identity


@pytest.mark.parametrize("width", [24, 51, 52])
@pytest.mark.parametrize("enabled", [False, True])
def test_public_option_preserves_default_and_small_or_odd_dispatch(width, enabled):
    solver = BlockLOBPCG(
        Identity(3 * width + 7, wp.get_device()),
        4,
        width=width,
        reuse_operator_images=True,
        search_bucket_size=4,
        compact_projected_graphs=enabled,
    )
    selected = []
    for branch in [solver, *solver._search.buckets]:
        for projected in (branch.small, branch.metric):
            inner = projected._global_jacobi
            expected = enabled and inner is not None and 96 <= inner.n <= 1024
            assert isinstance(inner, CompactJacobi) == expected
            if expected:
                assert inner.rounds_per_loop == 16
                selected.append(inner)
    if enabled and width >= 51 and wp.get_device().is_cuda:
        assert selected
        assert len({x.round_index.ptr for x in selected}) == len(selected)


def test_coupled_mass_constraints_clusters_and_changed_replay(tmp_path):
    n, k, width = 192, 6, 52

    def fixture(seed):
        rng = np.random.default_rng(seed)
        lower = np.eye(n) + np.diag(np.linspace(0.05, 0.25 + seed * 0.001, n - 1), -1)
        mass = lower @ lower.T
        q = la.qr(rng.normal(size=(n, n)))[0]
        modes = la.solve_triangular(lower.T, q, lower=False)
        values = np.repeat(np.arange(n // 2, dtype=float), 2)
        a = mass @ (modes * values) @ modes.T @ mass
        constraints = (modes[:, :2] @ np.array([[2.0, 0.4], [0.0, 3.0]])).T.copy()
        return a, mass, modes, values, constraints

    a, mass, modes, values, constraints = fixture(19)
    op, metric, pre = [
        SparseOperator(matrix(sp.csr_matrix(x))) for x in (a, mass, la.inv(a + mass))
    ]
    basis = wp.array(constraints, dtype=wp.float64)
    solver = BlockLOBPCG(
        op,
        k,
        width=width,
        mass=metric,
        preconditioner=pre,
        constraints=basis,
        normalize_constraints=True,
        tol=1e-10,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
        reuse_operator_images=True,
        search_bucket_size=4,
        keep_history=True,
        compact_projected_graphs=True,
    )
    assert isinstance(solver.small._global_jacobi, CompactJacobi)
    graph = solver.capture(200)
    for seed in [19, 31, 19]:
        a, mass, modes, values, constraints = fixture(seed)
        for target, host in [(op, a), (metric, mass), (pre, la.inv(a + mass))]:
            csr = sp.csr_matrix(host)
            np.testing.assert_array_equal(target.matrix.columns.numpy(), csr.indices)
            target.matrix.values.assign(csr.data)
        basis.assign(constraints)
        wp.capture_launch(graph)
        result, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        np.testing.assert_allclose(result, values[2 : 2 + k], atol=1e-10)
        assert np.linalg.norm(vectors.T @ mass @ vectors - np.eye(k)) < 1e-10
        assert np.linalg.norm(a @ vectors - (mass @ vectors) * result) < 1e-8
        wanted = modes[:, 2 : 2 + k]
        assert np.linalg.norm(vectors - wanted @ (wanted.T @ mass @ vectors)) < 1e-8
        assert np.linalg.norm(constraints @ mass @ vectors) < 1e-9
        np.testing.assert_array_equal(basis.numpy(), constraints)
    dot = tmp_path / "compact-block.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    content = dot.read_text()
    assert "HtoD" not in content and "DtoH" not in content and "HOST" not in content
