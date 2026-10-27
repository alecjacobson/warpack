"""The native handle and Warp operations must use the same setup stream."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from tests.test_solver import sparse
from warpack import CuDSSInverse, KrylovSchur


def test_wrong_stream_rejected_before_native_or_warp_dispatch(monkeypatch):
    inverse = object.__new__(CuDSSInverse)
    inverse.n, inverse.width, inverse.device = 8, 1, "device sentinel"
    inverse.workspace = SimpleNamespace(stream=SimpleNamespace(cuda_stream=17))
    inverse.refinement_steps = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched stream reached native solve")

    inverse.solver = SimpleNamespace(solve=forbidden)
    rows = SimpleNamespace(
        ndim=2, shape=(1, 8), dtype=wp.float64, device=inverse.device, is_contiguous=True
    )
    monkeypatch.setattr(wp, "get_stream", lambda _: SimpleNamespace(cuda_stream=23))
    with pytest.raises(RuntimeError, match="construction stream"):
        inverse.apply(rows, rows)


def test_nondefault_stream_capture_and_changed_rhs():
    device = wp.get_device("cuda:0")
    stream = wp.Stream(device)
    inverse = graph = solver = None
    try:
        with wp.ScopedStream(stream):
            host = np.diag(np.arange(1.0, 33.0)) + 0.1 * (np.eye(32, k=1) + np.eye(32, k=-1))
            inverse = CuDSSInverse(sparse(host), 1)
            x = wp.ones((1, 32), dtype=wp.float64, device=device)
            y = wp.empty_like(x)
            inverse.apply(x, y)
            wp.synchronize_stream(stream)
            with wp.ScopedCapture(stream=stream) as capture:
                inverse.apply(x, y)
            graph = capture.graph
            for seed in [7, 19, 7]:
                rhs = np.random.default_rng(seed).normal(size=x.shape)
                x.assign(rhs)
                wp.capture_launch(graph, stream=stream)
                np.testing.assert_allclose(y.numpy(), np.linalg.solve(host, rhs.T).T, atol=1e-12)
                np.testing.assert_array_equal(x.numpy(), rhs)
            graph = capture = None
            solver = KrylovSchur(inverse, 3, ncv=24, which="LA", tol=1e-11)
            graph = solver.capture(4, adaptive=False)
            wp.capture_launch(graph, stream=stream)
            assert solver.converged.numpy()[0] == 3
            np.testing.assert_allclose(
                1 / solver.eigenvalues.numpy(), np.linalg.eigvalsh(host)[:3], atol=1e-9
            )
            wp.synchronize_stream(stream)
        # Reject the default stream as well; returning to the owner stream works.
        with pytest.raises(RuntimeError, match="construction stream"):
            inverse.apply(x, y)
        # Native calls are stream-bound; a completed graph remains portable.
        wp.capture_launch(graph)
        assert solver.converged.numpy()[0] == 3
        np.testing.assert_allclose(
            1 / solver.eigenvalues.numpy(), np.linalg.eigvalsh(host)[:3], atol=1e-9
        )
        with wp.ScopedStream(stream):
            inverse.apply(x, y)
            np.testing.assert_allclose(y.numpy(), np.linalg.solve(host, rhs.T).T, atol=1e-12)
            wp.synchronize_stream(stream)
    finally:
        graph = None
        if solver is not None:
            solver.graph = None
        if inverse is not None:
            inverse.close()
