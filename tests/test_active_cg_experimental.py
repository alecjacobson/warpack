"""Captured active-CG packing must reactivate rows and consume current inputs."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner as BatchedCGInverse
from warpack._active_cg import DeviceBucketedCG

pytestmark = pytest.mark.skipif(
    not all(hasattr(linear.LinearOperator, attr) for attr in ("batch_offsets", "max_batch_length")),
    reason="upstream batched CG required",
)


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize(
    "matrix_layout, factor_layout, use_factor",
    [
        ("rows", "rows", False),
        ("rows", "rows", True),
        ("rows", "columns", True),
        ("columns", "rows", True),
        ("columns", "columns", True),
    ],
)
def test_active_bucket_reactivation_matrix_refresh_and_nested_graph(
    dtype, block, in_place, tmp_path, matrix_layout, factor_layout, use_factor
):
    n, width = 45, 7
    a = sp.diags(
        [-0.4 * np.ones(n - 1), np.linspace(2, 5, n), -0.4 * np.ones(n - 1)], [-1, 0, 1]
    ).tocsr()
    if block == 1:
        wa = matrix(a)
    else:
        b = a.tobsr(blocksize=(3, 3))
        wa = ws.bsr_from_triplets(
            n // 3,
            n // 3,
            wp.array(np.repeat(np.arange(n // 3), np.diff(b.indptr)), dtype=wp.int32),
            wp.array(b.indices, dtype=wp.int32),
            wp.array(b.data, dtype=wp.mat33d),
        )
    values = wa.values.numpy().copy()
    tol = 1e-5 if dtype == wp.float32 else 1e-12
    pre = linear.FSAI(wa, max_row_size=8, factor_dtype=wp.float32) if use_factor else None
    inverse = BatchedCGInverse(
        wa,
        width,
        tol=tol,
        maxiter=50,
        working_dtype=dtype,
        preconditioner=pre,
        fsai_layout=factor_layout,
        matrix_layout=matrix_layout,
    )
    if inverse.requires_static_outer_capture:
        pytest.skip("upstream CG lacks preallocated loop control")
    errors = wp.zeros(width, dtype=wp.float64)
    solver = DeviceBucketedCG(inverse, errors, tolerance=1e-9, bucket_size=3)
    # Packing and adapter scratch share backing; upstream internal CG storage
    # and publicly requested for_width adapters retain independent buffers.
    rhs_store, out_store = solver._packing_storage
    assert rhs_store.shape == out_store.shape == (width, n)
    assert rhs_store.ptr != out_store.ptr
    assert {rhs.ptr for rhs in solver.rhs} == {rhs_store.ptr}
    assert {out.ptr for out in solver.outputs} == {out_store.ptr}
    assert {state.b.ptr for state in solver.states} == {inverse.b.ptr}
    assert {state.x.ptr for state in solver.states} == {inverse.x.ptr}
    for attr in ("_r_and_z_buf", "_p_and_Ap"):
        assert len({getattr(state.state, attr).ptr for state in solver.states}) == len(
            solver.states
        )
    for attr in ("operator", "preconditioner"):
        operations = [getattr(state, attr) for state in solver.states]
        if operations[0] is None:
            continue
        if operations[0].coalesced is not None:
            for name in ("first", "second"):
                arrays = [getattr(op.coalesced, name) for op in operations]
                assert len({array.ptr for array in arrays}) == 1
                assert all(array.is_contiguous for array in arrays)
        else:
            for name in ("tmp", "out"):
                arrays = [getattr(op, name, None) for op in operations]
                if arrays[0] is not None:
                    assert len({array.ptr for array in arrays}) == 1
    independent = inverse.for_width(3)
    assert independent.b.ptr != inverse.b.ptr and independent.x.ptr != inverse.x.ptr
    if use_factor:
        assert independent.single_preconditioner is inverse.single_preconditioner
    for attr in ("operator", "preconditioner"):
        public, original = getattr(independent, attr), getattr(inverse, attr)
        if public is None:
            continue
        if public.coalesced is not None:
            assert public.coalesced.first.ptr != original.coalesced.first.ptr
            assert public.coalesced.second.ptr != original.coalesced.second.ptr
        else:
            assert public.tmp.ptr != original.tmp.ptr
            if getattr(public, "out", None) is not None:
                assert public.out.ptr != original.out.ptr
    x = wp.zeros((width, n), dtype=wp.float64)
    y = x if in_place else wp.empty_like(x)
    solver.apply(x, y)
    running = wp.zeros(1, dtype=wp.int32)
    wp.synchronize()

    def body():
        solver.apply(x, y)
        running.zero_()

    with wp.ScopedCapture() as captured:
        solver.reset_stats()
        running.fill_(1)
        wp.capture_while(running, body)
    rng = np.random.default_rng(7841)
    for iteration, count in enumerate((0, 1, 4, 7, 3, 6, 0, 2, 7)):
        scale = 1 + 0.1 * iteration
        wa.values.assign(scale * values)
        active = np.sort(rng.permutation(width)[:count])
        mask = np.zeros(width, dtype=bool)
        mask[active] = True
        error = np.where(mask, 1e-3, 1e-12)
        errors.assign(error)
        rhs = rng.normal(size=(width, n)) * np.logspace(-6, 6, width)[:, None]
        x.assign(rhs)
        wp.capture_launch(captured.graph)
        result = y.numpy()
        np.testing.assert_array_equal(result[~mask], 0)
        if count:
            expected = np.linalg.solve(scale * a.toarray(), rhs[mask].T).T
            rhs_norm = np.linalg.norm(rhs[mask], axis=1)
            limit = 2e-5 if dtype == wp.float32 else 2e-12
            assert np.max(np.linalg.norm(result[mask] - expected, axis=1) / rhs_norm) < limit
            assert (
                np.max(np.linalg.norm(result[mask] @ (scale * a.T) - rhs[mask], axis=1) / rhs_norm)
                < limit
            )
        assert solver.count.numpy()[0] == count
        bucket = next((w for w in (3, 6, 7) if w >= count), 0) if count else 0
        assert solver.stats.numpy()[2:].tolist() == [0, bucket]
        usage = solver.usage.numpy()
        assert usage[-1] == count
        assert usage[:-1].sum() == int(count > 0)
        if not in_place:
            np.testing.assert_array_equal(x.numpy(), rhs)
    dot = tmp_path / "active-cg.dot"
    wp.capture_debug_dot_print(captured.graph, str(dot))
    graph = dot.read_text()
    assert "DtoH" not in graph and "HtoD" not in graph and "HOST" not in graph


@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("kind", ["spd", "indefinite", "repeated", "rigid"])
def test_bucketed_eigensolver_keeps_complete_requested_space(metric, kind):
    import scipy.linalg as la

    from warpack import BlockLOBPCG, SparseOperator

    n, k, width = 64, 4, 8
    rng = np.random.default_rng(119)
    lower = np.eye(n)
    if metric:
        lower += np.diag(np.linspace(0.05, 0.3, n - 1), -1)
    mass = lower @ lower.T
    q = la.qr(rng.normal(size=(n, n)))[0]
    modes = la.solve_triangular(lower.T, q, lower=False)
    values = np.arange(1.0, n + 1)
    if kind == "indefinite":
        values -= 4
    if kind in ("repeated", "rigid"):
        values = np.ceil(values / 2)
    excluded = 6 if kind == "rigid" else 0
    values[:excluded] = 0
    a = mass @ (modes * values) @ modes.T @ mass
    op = SparseOperator(matrix(a))
    shifted = matrix(a + 5 * mass)
    inner = BatchedCGInverse(shifted, width, tol=0.01, maxiter=12, working_dtype=wp.float32)
    if inner.requires_static_outer_capture:
        pytest.skip("upstream CG lacks preallocated loop control")
    starts = wp.array(rng.normal(size=(width, n)), dtype=wp.float64)
    solver = BlockLOBPCG(
        op,
        k,
        width=width,
        mass=SparseOperator(matrix(mass)),
        initial_vectors=starts,
        initial_perturbation=1e-3,
        constraints=None
        if not excluded
        else wp.array(modes[:, :excluded].T.copy(), dtype=wp.float64),
        tol=1e-10,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
    )
    solver.preconditioner = DeviceBucketedCG(inner, solver.errors, solver.tol, bucket_size=2)
    graph = solver.capture(150)
    for changed in (False, True):
        current_values = values.copy()
        if changed:
            # A full old warm block omits the newly lowest modes. Keep a complete
            # repeated boundary cluster and leave rigid exclusions untouched.
            current_values[excluded : excluded + width] += 100
        current = mass @ (modes * current_values) @ modes.T @ mass
        for target, host in ((op.matrix, current), (shifted, current + 5 * mass)):
            csr = sp.csr_matrix(host)
            np.testing.assert_array_equal(target.columns.numpy(), csr.indices)
            target.values.assign(csr.data)
        wp.capture_launch(graph)
        out, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        order = np.argsort(current_values[excluded:])[:k] + excluded
        wanted = modes[:, order]
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        np.testing.assert_allclose(out, current_values[order], atol=1e-9)
        np.testing.assert_allclose(vectors.T @ mass @ vectors, np.eye(k), atol=1e-10)
        assert la.norm(current @ vectors - (mass @ vectors) * out, axis=0).max() < 2e-8
        assert la.norm(vectors - wanted @ (wanted.T @ mass @ vectors)) < 1e-7
        wp.copy(starts, solver.x)


def test_active_storage_sharing_preserves_custom_width_factory():
    calls = []

    class Custom(BatchedCGInverse):
        def for_width(self, width):
            calls.append(width)
            return super().for_width(width)

    base = Custom(matrix(sp.eye(12).tocsr()), 7, maxiter=2)
    owner = DeviceBucketedCG(base, wp.zeros(7, dtype=wp.float64), 1e-9, bucket_size=3)
    assert calls == [3, 6]
    assert len({adapter.b.ptr for adapter in owner.states}) == 3
