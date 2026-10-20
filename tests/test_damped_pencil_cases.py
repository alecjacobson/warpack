"""CPU-only independent references for spatially consistent damped pencils."""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment

from benchmarks.applications.damped import quality
from benchmarks.applications.damped_pencil import consistent_chain


@pytest.mark.parametrize("kind", ["nonproportional", "gyroscopic"])
def test_consistent_reference_full_dense_pencil(kind):
    case = consistent_chain(nodes=12, k=8, kind=kind)
    a, b = case.a.toarray(), case.b.toarray()
    assert np.count_nonzero(b - np.diag(np.diag(b))) > 0
    assert la.eigvalsh(b).min() > 0
    expected = la.eigvals(a, b)
    rows, cols = linear_sum_assignment(abs(expected[:, None] - case.full_spectrum))
    assert abs(expected[rows] - case.full_spectrum[cols]).max() < 1e-10
    assert quality(case, case.values, case.right)["accepted"]
    np.testing.assert_allclose(case.left.conj().T @ (b @ case.right), np.eye(case.k), atol=1e-12)
    np.testing.assert_allclose(
        case.left.conj().T @ a,
        case.values[:, None] * (case.left.conj().T @ b),
        atol=1e-11,
    )
    wrong = case.right.copy()
    wrong[:, -1] = wrong[:, 0]
    assert not quality(case, case.values, wrong)["accepted"]
    # Changing only M must invalidate the original pencil and quadratic gates.
    case.mass *= 1.2
    case.b.data[case.b.indptr[len(a) // 2] :] *= 1.2
    assert not quality(case, case.values, case.right)["accepted"]
