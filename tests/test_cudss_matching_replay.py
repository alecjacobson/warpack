"""Read back the actual cuDSS policy and check original systems after replay."""

import ctypes as ct

import numpy as np
import pytest
import warp as wp

from tests.test_solver import sparse
from warpack import CuDSSInverse


def assert_native_matching(inverse, expected):
    from warp_cudss._bindings import CUDSS_CONFIG_MATCHING_ALG, _check

    value, size = ct.c_int(-1), ct.c_size_t()
    solver = inverse.solver
    _check(
        solver._lib.cudssConfigGet(
            solver._config,
            CUDSS_CONFIG_MATCHING_ALG,
            ct.byref(value),
            ct.sizeof(value),
            ct.byref(size),
        ),
        "cudssConfigGet(MATCHING_ALG)",
    )
    assert size.value == ct.sizeof(value)
    assert value.value == expected


@pytest.mark.parametrize("mtype,matching", [("spd", False), ("general", False), ("general", True)])
@pytest.mark.parametrize("width", [1, 3])
def test_native_matching_and_changed_rhs_graph(mtype, matching, width):
    n = 64
    diagonal = 4 * np.ones(n)
    if mtype == "general":
        diagonal[::2] *= -1
    a = np.diag(diagonal) + 0.35 * (np.eye(n, k=1) + np.eye(n, k=-1))
    inverse = CuDSSInverse(sparse(a), width, mtype=mtype, matching=matching)
    captured = None
    try:
        assert_native_matching(inverse, 5 if matching else 0)
        rhs = wp.zeros((width, n), dtype=wp.float64, device="cuda:0")
        out = wp.empty_like(rhs)
        inverse.apply(rhs, out)
        wp.synchronize_device("cuda:0")
        with wp.ScopedCapture("cuda:0") as captured:
            inverse.apply(rhs, out)
        for seed in [9, 27, 9]:
            values = np.random.default_rng(seed).normal(size=(width, n))
            rhs.assign(values)
            out.fill_(float("nan"))
            wp.capture_launch(captured.graph)
            result = out.numpy()
            np.testing.assert_allclose(
                result, np.linalg.solve(a, values.T).T, rtol=1e-11, atol=1e-12
            )
            residual = np.linalg.norm(result @ a.T - values, axis=1) / np.linalg.norm(
                values, axis=1
            )
            assert residual.max() < 1e-12
            np.testing.assert_array_equal(rhs.numpy(), values)
    finally:
        if captured is not None:
            captured.graph = None
        inverse.close()


def test_native_matching_disabled_for_symmetric_indefinite_eager_solve():
    # AUTO pivoting has a different capture contract in cuDSS 1.0. This test
    # checks the independent matching setting without overriding that policy.
    a = np.diag(np.linspace(-7.5, 7.5, 16)) + 0.1 * (np.eye(16, k=1) + np.eye(16, k=-1))
    inverse = CuDSSInverse(sparse(a), 1, mtype="symmetric", matching=False)
    try:
        assert_native_matching(inverse, 0)
        values = np.random.default_rng(34).normal(size=(1, 16))
        rhs = wp.array(values, dtype=wp.float64, device="cuda:0")
        out = wp.empty_like(rhs)
        inverse.apply(rhs, out)
        np.testing.assert_allclose(
            out.numpy(), np.linalg.solve(a, values.T).T, rtol=1e-11, atol=1e-12
        )
    finally:
        inverse.close()
