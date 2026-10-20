"""Residual-only checks must not accept wrong or incomplete LM spectra."""

import numpy as np

from benchmarks.upstream.general_reference import GeneralReference


def test_boundary_tie_preserves_every_strictly_larger_value():
    values = np.array([10.0, 3.0 + 4j, 3.0 - 4j, 1.0])
    reference = GeneralReference(np.diag(values), 2)
    for indices in ([0, 1], [0, 2]):
        assert reference.check(values[indices], np.eye(4)[:, indices])["accepted"]
    wrong = reference.check(values[[1, 2]], np.eye(4)[:, [1, 2]])
    assert wrong["max_absolute_residual"] == 0
    assert wrong["spectrum_error"] == 0
    assert not wrong["required_spectrum_covered"]
    assert not wrong["accepted"]


def test_duplicate_zero_vectors_and_wrong_end_fail():
    values = np.arange(1.0, 6)
    reference = GeneralReference(np.diag(values), 2)
    assert reference.check(values[-2:], np.eye(5)[:, -2:])["accepted"]
    assert not reference.check(values[:2], np.eye(5)[:, :2])["accepted"]
    assert not reference.check(values[-2:], np.zeros((5, 2)))["accepted"]
    assert not reference.check(values[[4, 4]], np.eye(5)[:, [4, 4]])["accepted"]
