import numpy as np
import pytest
import warp as wp
import warp.sparse as ws

from warpack import PartialSVD


@pytest.mark.parametrize("shape", [(50, 30), (25, 45)])
def test_rectangular_svd(shape):
    rng = np.random.default_rng(13)
    a = rng.normal(size=shape)
    r, c = np.nonzero(a)
    mat = ws.bsr_from_triplets(
        *shape,
        wp.array(r, dtype=wp.int32),
        wp.array(c, dtype=wp.int32),
        wp.array(a[r, c], dtype=wp.float64),
    )
    s = PartialSVD(mat, 5, tol=1.0e-11)
    wp.capture_launch(s.capture(100))
    sigma = s.singular_values.numpy()
    u = s.u.numpy().T
    v = s.v.numpy().T
    np.testing.assert_allclose(sigma, np.linalg.svd(a, compute_uv=False)[:5], atol=1.0e-9)
    np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1.0e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1.0e-10)
    assert np.linalg.norm(a @ v - u * sigma, axis=0).max() < 1.0e-9
    assert np.linalg.norm(a.T @ u - v * sigma, axis=0).max() < 1.0e-9


@pytest.mark.parametrize("shape", [(1031, 20), (20, 1031), (2053, 20)])
def test_long_snapshot_rows_repeated_singular_subspace(shape):
    from benchmarks.upstream.adapters import matrix

    rng = np.random.default_rng(131)
    rank = min(shape)
    left = np.linalg.qr(rng.normal(size=(shape[0], rank)))[0]
    right = np.linalg.qr(rng.normal(size=(shape[1], rank)))[0]
    spectrum = np.r_[np.full(3, 10.0), 8.0, np.linspace(6, 0.5, rank - 6), 0.0, 0.0]
    a = (left * spectrum) @ right.T
    solver = PartialSVD(matrix(a), 4, ncv=24, tol=1e-11)
    graph = solver.capture(100)
    for _ in range(2):
        wp.capture_launch(graph)
        values, u, v = solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T
        np.testing.assert_allclose(values, spectrum[:4], atol=1e-9, rtol=0)
        np.testing.assert_allclose(u.T @ u, np.eye(4), atol=1e-10, rtol=0)
        np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-10, rtol=0)
        assert np.linalg.norm(a @ v - u * values, axis=0).max() < 1e-9
        assert np.linalg.norm(a.T @ u - v * values, axis=0).max() < 1e-9
        assert np.linalg.norm(u - left[:, :4] @ (left[:, :4].T @ u)) < 1e-9
        assert np.linalg.norm(v - right[:, :4] @ (right[:, :4].T @ v)) < 1e-9
        assert solver.count_left.numpy()[0] == 4
        assert solver.count_right.numpy()[0] == 4


@pytest.mark.parametrize("variant", ["tall", "wide", "clustered", "wave"])
def test_pod_fixture_reference_and_selection_gate(variant):
    from benchmarks.applications.pod_compare import snapshots
    from benchmarks.applications.runner import accuracy

    problem = snapshots(101, variant)
    u, values, vt = np.linalg.svd(problem.a.toarray(), full_matrices=False)
    result = accuracy(problem, values[:4], u[:, :4], vt[:4].T)
    assert result["accepted"]
    assert result["left_subspace_error"] < 1e-12
    assert result["right_subspace_error"] < 1e-12
    assert result["reconstruction_energy_error"] < 1e-12
    wrong = accuracy(problem, values[4:8], u[:, 4:8], vt[4:8].T)
    assert wrong["residual"] < 1e-12
    assert not wrong["accepted"]
