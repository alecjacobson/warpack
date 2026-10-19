"""Normal-product POD: original equations, multiplicities and rank limits."""

import re

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.pod_compare import snapshots
from benchmarks.applications.runner import accuracy
from benchmarks.upstream.adapters import matrix
from warpack import NormalSVD
from warpack.normal_svd import NormalOperator


@pytest.mark.parametrize("transpose", [False, True])
def test_normal_operator_matrix_free_products_replay(transpose):
    a = np.random.default_rng(91).normal(size=(5003, 12))
    a[:, 2] = 0
    if transpose:
        a = a.T.copy()
    op = NormalOperator(matrix(a), 3)
    x = wp.zeros((3, op.n), dtype=wp.float64)
    y = wp.empty_like(x)
    op.apply(x, y)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        op.apply(x, y)
    for seed in (17, 18):
        host = np.random.default_rng(seed).normal(size=x.shape)
        x.assign(host)
        wp.capture_launch(capture.graph)
        expected = host @ (a.T @ a if op.tall else a @ a.T)
        np.testing.assert_allclose(y.numpy(), expected, atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("variant", ["tall", "wide", "clustered", "wave"])
@pytest.mark.parametrize("ncv", [40, 48])
def test_normal_svd_complete_nonzero_triplets(variant, ncv, tmp_path):
    p = snapshots(101, variant, modes=20)
    solver = NormalSVD(matrix(p.a), p.k, ncv=ncv, tol=1e-11)
    graph = solver.capture(100)
    for _ in range(2):
        wp.capture_launch(graph)
        result = accuracy(p, solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T)
        assert result["accepted"], result
        assert solver.core.status.numpy()[0] == 0
        assert solver.count_left.numpy()[0] == p.k
        assert solver.count_right.numpy()[0] == p.k
    dot = tmp_path / "normal.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("tiny", [0.0, 1e-9])
def test_normal_svd_rejects_unresolved_requested_triplets(transpose, tiny):
    a = np.zeros((30, 12))
    a[:4, :4] = np.diag([10, 5, 2, tiny])
    if transpose:
        a = a.T.copy()
    solver = NormalSVD(matrix(a), 4, ncv=12, tol=1e-10)
    wp.capture_launch(solver.capture(5))
    assert solver.core.status.numpy()[0] == 2
    assert solver.count_left.numpy()[0] < 4
    assert solver.count_right.numpy()[0] < 4
    assert np.all(np.isfinite(solver.singular_values.numpy()))
    assert np.all(np.isfinite(solver.u.numpy()))
    assert np.all(np.isfinite(solver.v.numpy()))


@pytest.mark.parametrize("transpose", [False, True])
def test_normal_svd_replay_recovers_after_unresolved_input(transpose):
    a = np.zeros((30, 12))
    a[:4, :4] = np.diag([10, 5, 2, 1e-9])
    if transpose:
        a = a.T.copy()
    wa = matrix(a)
    solver = NormalSVD(wa, 4, ncv=12, tol=1e-11)
    graph = solver.capture(5)
    wp.capture_launch(graph)
    assert solver.core.status.numpy()[0] == 2
    wa.values.assign(np.array([10.0, 5.0, 2.0, 1.0]))
    wp.capture_launch(graph)
    np.testing.assert_allclose(solver.singular_values.numpy(), [10, 5, 2, 1], atol=1e-11)
    assert solver.core.status.numpy()[0] == 0
    assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 4


@pytest.mark.parametrize("variant", ["tall", "wide", "clustered"])
@pytest.mark.parametrize("ncv", [48, 96])
def test_normal_svd_wider_snapshot_space(variant, ncv, tmp_path):
    p = snapshots(257, variant, modes=12, columns=96)
    solver = NormalSVD(matrix(p.a), p.k, ncv=ncv, tol=1e-11)
    graph = solver.capture(100)
    for _ in range(2):
        wp.capture_launch(graph)
        check = accuracy(p, solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T)
        assert check["accepted"], check
        assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == p.k
        assert solver.core.status.numpy()[0] == 0
    dot = tmp_path / "normal-wide-snapshots.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1])
def test_normal_orthogonality_guard_is_captured_and_resets(bad):
    from warpack.normal_svd import check_orthogonality

    gram = wp.array(np.eye(3), dtype=wp.float64)
    errors = wp.zeros(2, dtype=wp.float64)
    status = wp.zeros(1, dtype=wp.int32)
    left, right = wp.zeros_like(status), wp.zeros_like(status)
    args = [gram, errors, 0, 1e-8, status, left, right]
    wp.launch(check_orthogonality, 1, args)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        wp.launch(check_orthogonality, 1, args)
    for value in (bad, 0.0, bad, 0.0):
        host = np.eye(3)
        host[0, 1] = value
        gram.assign(host)
        status.zero_()
        left.assign(np.array([3], dtype=np.int32))
        right.assign(np.array([3], dtype=np.int32))
        wp.capture_launch(capture.graph)
        invalid = not np.isfinite(value) or value > 1e-8
        assert status.numpy()[0] == (3 if invalid else 0)
        assert left.numpy()[0] == right.numpy()[0] == (0 if invalid else 3)
        assert errors.numpy()[0] == (float("inf") if not np.isfinite(value) else value)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_normal_invalid_orthogonality_tolerance(tolerance):
    with pytest.raises(ValueError, match="orthogonality_tol"):
        NormalSVD(None, 2, orthogonality_tol=tolerance)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("scale", [1e-3, 1.0])
def test_normal_svd_rejects_ill_conditioned_lift_and_recovers(transpose, scale, tmp_path):
    rng = np.random.default_rng(394)
    left = np.linalg.qr(rng.normal(size=(91, 32)))[0]
    right = np.linalg.qr(rng.normal(size=(32, 32)))[0]

    def fixture(tail):
        singular = np.r_[np.geomspace(1, tail, 8), np.geomspace(tail * 0.1, tail * 0.001, 24)]
        a = (left * (scale * singular)) @ right.T
        return a.T.copy() if transpose else a

    wa = matrix(fixture(2e-7))
    solver = NormalSVD(wa, 8, ncv=32, tol=1e-11)
    graph = solver.capture(3)
    for tail in (2e-7, 1e-3, 2e-7, 1e-3):
        a = fixture(tail)
        wa.values.assign(a.ravel())
        wp.capture_launch(graph)
        u, v = solver.u.numpy().T, solver.v.numpy().T
        errors = np.array([abs(u.T @ u - np.eye(8)).max(), abs(v.T @ v - np.eye(8)).max()])
        np.testing.assert_allclose(
            solver.orthogonality_errors.numpy(), errors, atol=2e-15, rtol=1e-5
        )
        if tail == 2e-7:
            assert errors.max() > solver.orthogonality_tol
            assert solver.core.status.numpy()[0] == 3
            assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 0
        else:
            assert errors.max() < 1e-8
            assert solver.core.status.numpy()[0] == 0
            assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 8
            values = solver.singular_values.numpy()
            reference = np.linalg.svd(a, compute_uv=False)[:8]
            np.testing.assert_allclose(values, reference, atol=1e-12, rtol=1e-10)
            assert np.linalg.norm(a @ v - u * values, axis=0).max() < 1e-11
            assert np.linalg.norm(a.T @ u - v * values, axis=0).max() < 1e-11
    dot = tmp_path / "normal-conditioning.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("transpose", [False, True])
def test_public_normal_svd_eager_observes_current_values(transpose):
    host = np.random.default_rng(831).normal(size=(51, 19))
    if transpose:
        host = host.T.copy()
    a = matrix(host)
    solver = NormalSVD(a, 5, ncv=19, tol=1e-11)
    reference = np.linalg.svd(host, compute_uv=False)[:5]
    for scale in (1.0, 2.0, 1.0):
        a.values.assign((scale * host).ravel())
        assert solver.solve(iterations=2) is solver
        assert solver.status is solver.core.status
        assert solver.status.numpy()[0] == 0
        assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 5
        np.testing.assert_allclose(solver.singular_values.numpy(), scale * reference, atol=1e-11)
        u, v = solver.u.numpy().T, solver.v.numpy().T
        values = solver.singular_values.numpy()
        assert np.linalg.norm(scale * host @ v - u * values, axis=0).max() < 1e-10
        assert np.linalg.norm(scale * host.T @ u - v * values, axis=0).max() < 1e-10
        np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1e-11)
        np.testing.assert_allclose(v.T @ v, np.eye(5), atol=1e-11)
