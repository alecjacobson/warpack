"""Indefinite foundation modes with upstream GMRES and a separate SPD proxy."""

import re

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.applications.negative_foundation import make_negative_foundation
from benchmarks.experimental.interior_gmres import assess_trial, gmres_inverse
from benchmarks.upstream.adapters import matrix
from warpack import GeneralizedKrylovSchur, SparseOperator


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("preconditioner", ["none", "diag", "fsai"])
def test_negative_gmres_complete_requested_space(consistent, preconditioner, tmp_path):
    if preconditioner == "fsai" and not hasattr(linear, "FSAI"):
        pytest.skip("upstream FSAI is unavailable in stock Warp")
    case = make_negative_foundation(3, 6, "SM", consistent)
    p = case.problem
    proxy = p.a + (case.foundation + 1) * p.b
    assert la.eigvalsh(proxy.toarray()).min() > 0
    assert la.eigvalsh(p.a.toarray()).min() < 0
    a, b = matrix(p.a), matrix(p.b)
    inverse = gmres_inverse(
        a,
        matrix(proxy),
        restart=p.n,
        maxiter=3 * p.n,
        inner_tol=1e-13,
        preconditioner=preconditioner,
        fsai_width=16,
        adaptive_capture=False,
    )
    solver = GeneralizedKrylovSchur(
        SparseOperator(a), SparseOperator(b), inverse, p.k, target=0, ncv=p.n, tol=1e-11
    )
    graph = solver.capture(3)
    for _ in range(2):
        inverse.reset_stats()
        wp.capture_launch(graph)
        q = assess_trial(p, solver, inverse, "gmres")
        assert q["accepted"], q
        assert q["physical_accepted"] and q["outer_stopping_accepted"]
        assert q["inner_solve_accepted"]
        assert inverse.stats.numpy()[3] > 0
        assert np.isfinite(inverse.residual_maxima.numpy()).all()
    dot = tmp_path / "negative-gmres.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert not re.search(r'label="\{\s*HOST\b', text)
    assert "HtoD" not in text and "DtoH" not in text


@pytest.mark.parametrize("consistent", [False, True])
def test_negative_gmres_exhaustion_is_rejected(consistent):
    p = make_negative_foundation(3, 6, "SM", consistent).problem
    a, b = matrix(p.a), matrix(p.b)
    inverse = gmres_inverse(
        a,
        None,
        restart=1,
        maxiter=1,
        inner_tol=1e-13,
        preconditioner="none",
        adaptive_capture=False,
    )
    solver = GeneralizedKrylovSchur(
        SparseOperator(a), SparseOperator(b), inverse, p.k, target=0, ncv=p.n, tol=1e-11
    )
    graph = solver.capture(1)
    inverse.reset_stats()
    wp.capture_launch(graph)
    q = assess_trial(p, solver, inverse, "gmres")
    assert inverse.stats.numpy()[2] > 0
    assert not q["inner_solve_accepted"] and not q["accepted"]
    # Even a physically exact, full-count outer result must not conceal the
    # actual inner failures above. This tests the benchmark gate independently
    # of outer nonconvergence, without using reference vectors in a solve.
    wp.copy(solver.eigenvalues, wp.array(p.expected, dtype=wp.float64))
    wp.copy(
        solver.eigenvectors, wp.array(np.ascontiguousarray(p.requested_basis.T), dtype=wp.float64)
    )
    solver.status.zero_()
    solver.converged.fill_(p.k)
    q = assess_trial(p, solver, inverse, "gmres")
    assert q["physical_accepted"] and q["outer_stopping_accepted"], q
    assert not q["inner_solve_accepted"] and not q["accepted"]
