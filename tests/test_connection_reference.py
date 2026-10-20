"""Explicit complex ARPACK metric recovery never supplies missing directions."""

import numpy as np
import pytest
import scipy.linalg as la

from benchmarks.applications.connection import make_connection_case
from benchmarks.applications.connection_backends import metric_ritz
from benchmarks.applications.runner import accuracy


def test_nonorthogonal_repeated_basis_recovered():
    p = make_connection_case(side=5, modes=6, copies=2).problem
    basis = p.requested_basis
    rng = np.random.default_rng(254)
    mixing = np.eye(p.k) + 0.1 * (rng.normal(size=(p.k, p.k)) + 1j * rng.normal(size=(p.k, p.k)))
    vectors = basis @ mixing
    assert np.max(abs(vectors.conj().T @ (p.b @ vectors) - np.eye(p.k))) > 0.1
    values, recovered = metric_ritz(p, vectors)
    check = accuracy(p, values, recovered)
    assert check["accepted"], check


def test_wrong_eigenspace_still_fails_after_metric_recovery():
    p = make_connection_case(side=5, modes=4).problem
    _, basis = la.eigh(p.a.toarray(), p.b.toarray())
    values, recovered = metric_ritz(p, basis[:, -p.k :])
    check = accuracy(p, values, recovered)
    assert check["residual"] < 1e-10
    assert check["orthogonality"] < 1e-10
    assert not check["accepted"]


def test_rank_deficient_returned_space_rejected():
    p = make_connection_case(side=5, modes=4).problem
    vectors = np.zeros((p.n, p.k), dtype=complex)
    with pytest.raises(la.LinAlgError):
        metric_ritz(p, vectors)
