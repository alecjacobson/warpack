"""Upstream BSR preconditioners keep a matrix dtype and scalar operator shape."""

import inspect

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator
from warpack.block import UpstreamPreconditioner


def block_fixture(block_size, blocks=12):
    rng = np.random.default_rng(839)
    values = []
    for i in range(blocks):
        q = la.qr(rng.normal(size=(block_size, block_size)))[0]
        spectrum = np.arange(block_size, dtype=float) * 20 + (i // 2 + 1)
        values.append(q @ np.diag(spectrum) @ q.T)
    dtype = wp.types.matrix(shape=(block_size, block_size), dtype=wp.float64)
    ids = wp.array(np.arange(blocks), dtype=wp.int32)
    device = ws.bsr_from_triplets(blocks, blocks, ids, ids, wp.array(np.array(values), dtype=dtype))
    return device, la.block_diag(*values)


def upstream(a, kind):
    if kind == "fsai":
        if not hasattr(linear, "FSAI"):
            pytest.skip("requires upstream Warp FSAI")
        return linear.FSAI(a, max_row_size=8, max_step_size=3)
    if "block_jacobi_sequential" not in inspect.getsource(linear.preconditioner):
        pytest.skip("requires upstream Warp block Jacobi")
    return linear.preconditioner(a, ptype="block_jacobi_sequential")


@pytest.mark.parametrize("kind", ["bj", "fsai"])
@pytest.mark.parametrize("block_size", [2, 3])
def test_upstream_bsr_preconditioner_changed_batched_rhs(kind, block_size, tmp_path):
    a, dense = block_fixture(block_size)
    pre = upstream(a, kind)
    assert pre.dtype != wp.float64 and pre.scalar_type == wp.float64
    wrapped = UpstreamPreconditioner(pre)
    assert wrapped.n == len(dense)
    rhs = wp.zeros((3, len(dense)), dtype=wp.float64)
    out = wp.empty_like(rhs)
    wrapped.apply(rhs, out)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wrapped.apply(rhs, out)
    for seed in (1, 2):
        host = np.random.default_rng(seed).normal(size=rhs.shape)
        rhs.assign(host)
        wp.capture_launch(capture.graph)
        expected = la.solve(dense, host.T, assume_a="pos").T
        np.testing.assert_allclose(out.numpy(), expected, atol=2e-12, rtol=2e-12)
        np.testing.assert_array_equal(rhs.numpy(), host)
    dot = tmp_path / "upstream-block.dot"
    wp.capture_debug_dot_print(capture.graph, str(dot))
    assert not any(token in dot.read_text() for token in ("HOST", "HtoD", "DtoH"))


@pytest.mark.parametrize("kind", ["bj", "fsai"])
def test_upstream_bsr_preconditioned_repeated_block_modes(kind):
    a, dense = block_fixture(3, blocks=16)
    expected, basis = la.eigh(dense)
    starts = wp.zeros((8, len(dense)), dtype=wp.float64)
    solver = BlockLOBPCG(
        SparseOperator(matrix(dense)),
        4,
        width=8,
        preconditioner=upstream(a, kind),
        initial_vectors=starts,
        tol=1e-11,
    )
    graph = solver.capture(100)
    for seed in (3, 4):
        starts.assign(np.random.default_rng(seed).normal(size=starts.shape))
        wp.capture_launch(graph)
        v, w = solver.eigenvectors.numpy().T, solver.eigenvalues.numpy()
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == 4
        np.testing.assert_allclose(w, expected[:4], atol=1e-10, rtol=0)
        assert la.norm(dense @ v - v * w, axis=0).max() < 1e-9
        assert la.norm(v - basis[:, :4] @ (basis[:, :4].T @ v)) < 1e-8
        np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-11)


@pytest.mark.parametrize("dtype", [wp.float32, wp.mat33f])
def test_upstream_block_adapter_rejects_non_fp64(dtype):
    pre = linear.LinearOperator((12, 12), dtype, wp.get_device(), lambda *args: None)
    with pytest.raises(ValueError, match="FP64"):
        UpstreamPreconditioner(pre)


def test_upstream_block_adapter_rejects_nonsquare():
    pre = linear.LinearOperator((12, 15), wp.mat33d, wp.get_device(), lambda *args: None)
    with pytest.raises(ValueError, match="square"):
        UpstreamPreconditioner(pre)


def test_upstream_adapter_preserves_scalar_matvec_metadata_contract():
    from types import SimpleNamespace

    def copy(x, y, z, alpha, beta):
        assert alpha == 1.0 and beta == 0.0
        wp.copy(z, x)

    pre = SimpleNamespace(shape=(6, 6), dtype=wp.float64, device=wp.get_device(), matvec=copy)
    wrapped = UpstreamPreconditioner(pre)
    rhs = wp.array(np.arange(12).reshape(2, 6), dtype=wp.float64)
    output = wp.empty_like(rhs)
    wrapped.apply(rhs, output)
    np.testing.assert_array_equal(output.numpy(), rhs.numpy())
