"""Complete normal-space projection retains original equations and requested spaces."""

import re

import numpy as np
import pytest
import warp as wp

from benchmarks.experimental.full_normal_basis import FullBasisNormalSVD
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("k", [4, 8])
def test_full_normal_projection_changed_values(transpose, repeated, k, tmp_path):
    rng = np.random.default_rng(485)
    u = np.linalg.qr(rng.normal(size=(73, 29)))[0]
    v = np.linalg.qr(rng.normal(size=(29, 29)))[0]
    values = np.geomspace(20, 0.05, 29)
    values[-2:] = 0
    if repeated:
        values[:3] = 20
    a = (u * values) @ v.T
    if transpose:
        a = a.T.copy()
    wa = matrix(a)
    solver = FullBasisNormalSVD(wa, k, ncv=29, tol=1e-11)
    graph = solver.capture(3)
    for host in (a, 2 * a, a * rng.uniform(0.9, 1.1, a.shape), a):
        wa.values.assign(host.ravel())
        wp.capture_launch(graph)
        ref_u, ref_s, ref_vt = np.linalg.svd(host, full_matrices=False)
        su, sv = solver.u.numpy().T, solver.v.numpy().T
        s = solver.singular_values.numpy()
        np.testing.assert_allclose(s, ref_s[:k], atol=1e-10, rtol=1e-11)
        assert np.linalg.norm(host @ sv - su * s, axis=0).max() < 1e-10
        assert np.linalg.norm(host.T @ su - sv * s, axis=0).max() < 1e-10
        np.testing.assert_allclose(su.T @ su, np.eye(k), atol=1e-10)
        np.testing.assert_allclose(sv.T @ sv, np.eye(k), atol=1e-10)
        for actual, reference in ((su, ref_u[:, :k]), (sv, ref_vt[:k].T)):
            assert np.linalg.norm(actual - reference @ (reference.T @ actual)) < 1e-9
        assert solver.status.numpy()[0] == 0
        assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == k
        assert solver.core.iterations.numpy()[0] == 1
        assert solver.core.recoveries.numpy()[0] == 0
        np.testing.assert_array_equal(wa.values.numpy(), host.ravel())
    dot = tmp_path / "full-normal.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "DtoH" not in text and "HtoD" not in text


@pytest.mark.parametrize("transpose", [False, True])
def test_full_normal_rejects_unresolved_values_and_recovers(transpose):
    rng = np.random.default_rng(439)
    u = np.linalg.qr(rng.normal(size=(43, 12)))[0]
    v = np.linalg.qr(rng.normal(size=(12, 12)))[0]

    def current(fourth):
        values = np.zeros(12)
        values[:4] = [10, 5, 2, fourth]
        a = (u * values) @ v.T
        return a.T.copy() if transpose else a

    wa = matrix(current(1e-9))
    solver = FullBasisNormalSVD(wa, 4, tol=1e-11)
    graph = solver.capture(3)
    for fourth in (1e-9, 1.0, 1e-9, 1.0):
        a = current(fourth)
        wa.values.assign(a.ravel())
        wp.capture_launch(graph)
        if fourth == 1e-9:
            assert solver.status.numpy()[0] == 2
            assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 0
        else:
            assert solver.status.numpy()[0] == 0
            assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 4
            w, su, sv = solver.singular_values.numpy(), solver.u.numpy().T, solver.v.numpy().T
            np.testing.assert_allclose(w, [10, 5, 2, 1], atol=1e-11)
            assert np.linalg.norm(a @ sv - su * w, axis=0).max() < 1e-10
            assert np.linalg.norm(a.T @ su - sv * w, axis=0).max() < 1e-10
            np.testing.assert_allclose(su.T @ su, np.eye(4), atol=1e-11)
            np.testing.assert_allclose(sv.T @ sv, np.eye(4), atol=1e-11)
