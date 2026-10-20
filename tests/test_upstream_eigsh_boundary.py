"""CuPy boundary validation and zero-restart initial-sweep semantics."""

import numpy as np
import pytest

from benchmarks.upstream.adapters import eigsh


@pytest.mark.parametrize(
    "keyword", [{"M": 1}, {"Minv": 1}, {"mode": "buckling"}, {"mode": "cayley"}]
)
def test_unsupported_keywords_reject_before_dtype_dispatch(keyword):
    pytest.importorskip("cupy")
    with pytest.raises(NotImplementedError):
        eigsh(np.eye(4, dtype=np.float32), k=1, **keyword)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_normal_mode_and_none_keywords_keep_the_supported_solve(dtype):
    cp = pytest.importorskip("cupy")
    a = np.diag(np.arange(1.0, 7.0)).astype(dtype)
    values, vectors = eigsh(a, k=2, which="SA", mode="normal", M=None, Minv=None)
    values, vectors = cp.asnumpy(values), cp.asnumpy(vectors)
    np.testing.assert_allclose(np.sort(values), [1.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(a @ vectors, vectors * values, atol=1e-10)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("width", [0, 1])
def test_clamped_width_zero_budget_keeps_exact_start(dtype, width):
    cp = pytest.importorskip("cupy")
    a = np.diag(np.arange(1.0, 33.0)).astype(dtype)
    start = np.zeros(32, dtype=dtype)
    start[-1] = 1.0
    before = start.copy()
    values, vectors = eigsh(a, k=1, v0=start, ncv=width, maxiter=0)
    values, vectors = cp.asnumpy(values), cp.asnumpy(vectors)
    np.testing.assert_allclose(values, [32.0], atol=1e-10)
    np.testing.assert_allclose(a @ vectors, vectors * values, atol=1e-10)
    np.testing.assert_array_equal(start, before)
