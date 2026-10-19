"""Normal-equation POD prototype: original equations, multiplicities and rank limits."""

import re

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.pod_compare import snapshots
from benchmarks.applications.runner import accuracy
from benchmarks.experimental.normal_svd import NormalOperator, NormalSVD
from benchmarks.upstream.adapters import matrix


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
