"""Fresh residuals and replay contracts for upstream Warp GMRES inverses."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp
import warp.sparse as ws
from warp.optim import linear

from benchmarks.upstream.adapters import matrix
from warpack import GMRESInverse


@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("nonsymmetric", [False, True])
@pytest.mark.parametrize("block", [1, 3])
def test_gmres_changed_matrix_and_rhs(preconditioned, in_place, nonsymmetric, block, tmp_path):
    n = 48
    a = sp.diags(
        [np.full(n - 1, 0.1), np.linspace(-3, 3, n), np.full(n - 1, 0.2 if nonsymmetric else 0.1)],
        [-1, 0, 1],
    ).tocsr()
    if block == 1:
        device = matrix(a)
        coefficients = a.data
    else:
        bsr = a.tobsr(blocksize=(3, 3))
        coefficients = bsr.data
        device = ws.bsr_from_triplets(
            n // 3,
            n // 3,
            wp.array(np.repeat(np.arange(n // 3), np.diff(bsr.indptr)), dtype=wp.int32),
            wp.array(bsr.indices, dtype=wp.int32),
            wp.array(coefficients, dtype=wp.mat33d),
        )
    pre = None
    if preconditioned:
        # A fixed positive scaling, not an SPD factorization of indefinite A.
        pre = linear.aslinearoperator(
            wp.array(1 / np.maximum(abs(a.diagonal()), 0.2), dtype=wp.float64)
        )
    inverse = GMRESInverse(device, restart=n, maxiter=3 * n, preconditioner=pre)
    b = wp.zeros((3, n), dtype=wp.float64)
    y = b if in_place else wp.empty_like(b)
    inverse.apply(b, y)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        inverse.reset_stats()
        inverse.apply(b, y)
    for scale in [1.0, 1.7]:
        rhs = np.random.default_rng(730).normal(size=b.shape)
        rhs[0] = 0
        b.assign(rhs)
        device.values.assign(coefficients * scale)
        wp.capture_launch(cap.graph)
        out = y.numpy()
        errors = la.norm(out @ (scale * a).T - rhs, axis=1)
        assert errors.max() < 1e-10
        np.testing.assert_allclose(out, la.solve(scale * a.toarray(), rhs.T).T, atol=1e-10)
        stats = inverse.stats.numpy()
        assert stats[2] == 0 and stats[3] == 3
        assert 0 < stats[1] <= 3 * n and stats[0] >= stats[1]
        np.testing.assert_allclose(inverse.residual_maxima.numpy()[0], errors.max(), atol=1e-13)
    dot = tmp_path / "gmres.dot"
    wp.capture_debug_dot_print(cap.graph, str(dot))
    assert not any(word in dot.read_text() for word in ["HOST", "DtoH", "HtoD"])


def test_gmres_exhaustion_is_visible():
    a = sp.diags(np.arange(-16.0, 16.0) + 0.2).tocsr()
    inverse = GMRESInverse(matrix(a), restart=1, maxiter=1)
    b = wp.ones((2, a.shape[0]), dtype=wp.float64)
    y = wp.empty_like(b)
    inverse.apply(b, y)
    assert inverse.stats.numpy().tolist() == [2, 1, 2, 2]
    assert inverse.residual_maxima.numpy()[1] > 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tol=-1),
        dict(tol=np.nan),
        dict(atol=np.inf),
        dict(tol=0, atol=0),
        dict(restart=0),
        dict(maxiter=-1),
        dict(restart=True),
        dict(maxiter=1.2),
    ],
)
def test_gmres_invalid_parameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        GMRESInverse(None, **kwargs)


@pytest.mark.parametrize(
    "name",
    ["clamped_consistent", "clamped_lumped", "heterogeneous", "disconnected", "repeated_bending"],
)
@pytest.mark.parametrize("fraction", [0.37, 0.001])
def test_gmres_interior_complete_requested_set(name, fraction, tmp_path):
    from benchmarks.applications.interior_fem import make_interior_case
    from benchmarks.applications.runner import accuracy, warp_output
    from warpack import GeneralizedKrylovSchur, SparseOperator

    p = make_interior_case(name, fraction=fraction).problem
    shifted = matrix(p.a - p.target * p.b)
    # Start with full GMRES subspaces to separate inverse capability from
    # restart/preconditioner limitations. Large restarted tests are benchmarks.
    inverse = GMRESInverse(shifted, tol=1e-13, restart=p.n, maxiter=3 * p.n)
    supports_nested = hasattr(inverse.state, "_cur_iter_and_condition")
    if supports_nested:
        inverse = GMRESInverse(
            shifted, tol=1e-13, restart=p.n, maxiter=3 * p.n, adaptive_capture=True
        )
    else:
        with pytest.raises(ValueError, match="preallocated loop control"):
            GMRESInverse(shifted, adaptive_capture=True)
    solver = GeneralizedKrylovSchur(
        SparseOperator(matrix(p.a)),
        SparseOperator(matrix(p.b)),
        inverse,
        p.k,
        ncv=min(48, p.n),
        tol=1e-9,
        target=p.target,
    )
    graph = solver.capture(12)
    for _ in range(2):
        inverse.reset_stats()
        wp.capture_launch(graph)
        check = accuracy(p, *warp_output(p, solver))
        assert check["accepted"], check
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == p.k
        assert inverse.stats.numpy()[3] > 0
    dot = tmp_path / "gmres-outer.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(word in dot.read_text() for word in ["HOST", "DtoH", "HtoD"])
