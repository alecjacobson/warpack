"""Device SVD starts: both orientations, two-sided accuracy and captured updates."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import PartialSVD


def check(solver, a, k):
    values = solver.singular_values.numpy()
    u, v = solver.u.numpy().T, solver.v.numpy().T
    np.testing.assert_allclose(values, np.linalg.svd(a, compute_uv=False)[:k], atol=1e-9, rtol=0)
    np.testing.assert_allclose(u.T @ u, np.eye(k), atol=1e-9, rtol=0)
    np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-9, rtol=0)
    assert np.linalg.norm(a @ v - u * values, axis=0).max() < 1e-9
    assert np.linalg.norm(a.T @ u - v * values, axis=0).max() < 1e-9
    assert solver.status.numpy()[0] == 0
    assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == k


@pytest.mark.parametrize("shape", [(45, 30), (30, 45)])
def test_svd_start_changed_matrix_and_extreme_scales(shape, tmp_path):
    rng = np.random.default_rng(701)
    rank, k = min(shape), 4
    left = np.linalg.qr(rng.normal(size=(shape[0], rank)))[0]
    right = np.linalg.qr(rng.normal(size=(shape[1], rank)))[0]
    spectra = [
        np.r_[10.0, 10.0, 10.0, 8.0, np.linspace(6.0, 0.5, rank - 6), 0.0, 0.0],
        np.r_[12.0, 11.0, 11.0, 9.0, np.linspace(7.0, 0.7, rank - 6), 0.0, 0.0],
    ]
    spectra[1][8] = 15.0  # A previously unrequested mode becomes the leader.
    matrices = [(left * values) @ right.T for values in spectra]
    a = matrix(matrices[0])
    start = wp.array(rng.normal(size=rank), dtype=wp.float64)
    solver = PartialSVD(a, k, ncv=24, tol=1e-11, initial_vector=start)
    graph = solver.capture(100)
    for index, scale in ((0, 1.0), (1, 1e300), (0, 1e-320), (1, 1.0)):
        dense = matrices[index]
        a.values.assign(dense.ravel())
        if index == 1 and scale == 1.0:
            # Reuse a computed old mode, never a validation reference vector.
            source = (solver.v if shape[0] >= shape[1] else solver.u).numpy()[0].copy()
        else:
            source = rng.uniform(0.5, 1.0, rank) * scale
        start.assign(source)
        wp.capture_launch(graph)
        check(solver, dense, k)
        np.testing.assert_array_equal(start.numpy(), source)
    dot = tmp_path / "svd-start.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert "HOST" not in text and "HtoD" not in text and "DtoH" not in text


@pytest.mark.parametrize("shape", [(12, 8), (8, 12)])
def test_svd_invalid_start_recovery_on_one_graph(shape):
    rng = np.random.default_rng(771)
    dense = rng.normal(size=shape)
    rank, k = min(shape), 3
    start = wp.ones(rank, dtype=wp.float64)
    solver = PartialSVD(matrix(dense), k, tol=1e-10, initial_vector=start)
    graph = solver.capture(20)
    for bad in (np.zeros(rank), np.full(rank, np.nan), np.full(rank, np.inf)):
        start.assign(bad)
        wp.capture_launch(graph)
        assert solver.status.numpy()[0] != 0
        assert solver.count_left.numpy()[0] == 0
        assert solver.count_right.numpy()[0] == 0
        assert solver.core.converged.numpy()[0] == 0
        start.assign(rng.normal(size=rank))
        wp.capture_launch(graph)
        check(solver, dense, k)


@pytest.mark.parametrize("shape", [(16, 10), (10, 16)])
def test_svd_start_null_space_does_not_lose_requested_multiplicity(shape):
    dense = np.zeros(shape)
    dense[np.arange(4), np.arange(4)] = [5.0, 5.0, 3.0, 1.0]
    source = np.zeros(min(shape))
    source[-1] = 1.0  # Exact null-space start; no reference top-mode information.
    solver = PartialSVD(
        matrix(dense),
        5,
        ncv=sum(shape),
        tol=1e-10,
        initial_vector=wp.array(source, dtype=wp.float64),
    )
    solver.solve(3)
    check(solver, dense, 5)


def test_svd_start_metadata_validation():
    a = matrix(np.eye(6, 4))
    invalid = [
        wp.ones(3, dtype=wp.float64),
        wp.ones(4, dtype=wp.float32),
        wp.ones((1, 4), dtype=wp.float64),
    ]
    for source in invalid:
        with pytest.raises(ValueError, match="shape.*FP64"):
            PartialSVD(a, 2, initial_vector=source)
    if a.device.is_cuda:
        with pytest.raises(ValueError, match="matrix device"):
            PartialSVD(a, 2, initial_vector=wp.ones(4, dtype=wp.float64, device="cpu"))
