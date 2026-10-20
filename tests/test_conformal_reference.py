"""Native ARPACK formulations agree with homogeneous QZ on singular masses."""

import numpy as np
import pytest

from benchmarks.applications.conformal import dense_reference, make_conformal_case, metrics
from benchmarks.applications.conformal_backends import scipy_setup


@pytest.mark.parametrize("original", [False, True])
@pytest.mark.parametrize("height", [0.0, 0.25])
def test_native_arpack_original_and_reciprocal(original, height):
    case = make_conformal_case(8, 6, height)
    w, v, _ = dense_reference(case)
    solve, close = scipy_setup(case, 32, original)
    try:
        values, vectors, _ = solve()
        result = metrics(case, values, vectors, (w, v))
        assert result["accepted"], result
    finally:
        close()


def test_translation_gauge_preserves_spectrum_and_centered_subspace():
    left = make_conformal_case(8, 6, 0.25, pin=0)
    right = make_conformal_case(8, 6, 0.25, pin=27)
    w, v, _ = dense_reference(left)
    wr, vr, _ = dense_reference(right)
    np.testing.assert_allclose(w, wr, atol=1e-13)
    assert metrics(right, wr, vr, (w, v))["accepted"]
