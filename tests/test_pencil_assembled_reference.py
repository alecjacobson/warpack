"""The stored-coefficient modal reference agrees with independent dense pencils."""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment

from benchmarks.applications.damped import quality
from benchmarks.applications.damped_pencil import consistent_chain
from benchmarks.experimental.pencil_assembled_reference import assembled_reference, modal_blocks


@pytest.mark.parametrize("kind", ["nonproportional", "gyroscopic"])
@pytest.mark.parametrize("nodes", [8, 32])
def test_assembled_reference_against_dense_and_original_equations(kind, nodes):
    original = consistent_chain(nodes, 8, kind)
    reference = assembled_reference(original)
    exact = la.eigvals(original.a.toarray(), original.b.toarray())
    exact = exact[np.argsort(abs(exact - original.target))[: original.k]]
    i, j = linear_sum_assignment(abs(reference.values[:, None] - exact[None, :]))
    np.testing.assert_allclose(reference.values[i], exact[j], atol=1e-10, rtol=1e-11)
    assert reference.a is original.a and reference.b is original.b
    assert quality(reference, reference.values, reference.right)["accepted"]
    np.testing.assert_allclose(
        reference.left.conj().T @ (original.b @ reference.right), np.eye(original.k), atol=1e-10
    )
    left_residual = (
        original.a.T @ reference.left - (original.b.T @ reference.left) * reference.values.conj()
    )
    assert la.norm(left_residual) / la.norm(reference.left) < 1e-10


def test_nonuniform_coefficients_are_rejected():
    case = consistent_chain(8, 6)
    changed = case.stiffness.copy()
    changed[4, 4] += 0.1
    with pytest.raises(ValueError, match="identical"):
        modal_blocks(changed, case.nodes)


def test_mismatched_linearization_is_rejected():
    case = consistent_chain(8, 6)
    case.a[0, 16] = 2.0
    with pytest.raises(ValueError, match="original consistent-chain"):
        assembled_reference(case)
