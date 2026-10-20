"""Distinct algebraic/magnitude targets of an indefinite 3D FEM pencil."""

import re

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.applications.negative_foundation import make_negative_foundation
from benchmarks.applications.runner import accuracy, warp_setup


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("selection", ["SA", "SM", "LM"])
@pytest.mark.parametrize("modes", [3, 6])
def test_negative_foundation_analytic_reference(consistent, selection, modes):
    case = make_negative_foundation(4, modes, selection, consistent)
    p = case.problem
    a, b = p.a.toarray(), p.b.toarray()
    w, v = la.eigh(a, b)
    np.testing.assert_allclose(w, case.full_spectrum, atol=3e-11, rtol=1e-12)
    ranking = np.argsort(w if selection == "SA" else abs(w) * (1 if selection == "SM" else -1))
    selected = ranking[: p.k]
    result = accuracy(p, w[selected], v[:, selected])
    assert result["accepted"], result
    assert p.k >= modes and p.k % 2 == 0
    assert w.min() < 0 < w.max() and abs(w).min() > 1e-8
    assert np.min(la.eigvalsh(b)) > 0
    if selection == "SA":
        assert np.max(p.expected) < 0
        assert np.min(la.eigvalsh(a - p.target * b)) > 0
    elif selection == "SM":
        assert np.min(p.expected) < 0 < np.max(p.expected)
    else:
        assert case.largest_magnitude_is_largest_algebraic
        np.testing.assert_allclose(np.sort(p.expected), w[-p.k :], atol=3e-11)


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize(
    "selection,method",
    [("SA", "krylov"), ("SM", "krylov"), ("LM", "krylov"), ("SA", "cg"), ("LM", "cg")],
)
def test_negative_foundation_captured_selection(consistent, selection, method, tmp_path):
    case = make_negative_foundation(3, 6, selection, consistent)
    p = case.problem
    solver, inverse = warp_setup(p, method=method, ncv=p.n, tol=1e-10, cg_tol=1e-13)
    try:
        graph = solver.capture(3)
        for _ in range(2):
            wp.capture_launch(graph)
            result = accuracy(p, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
            assert result["accepted"], result
            assert solver.status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == p.k
        dot = tmp_path / "negative-foundation.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        graph_text = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', graph_text) and "DtoH" not in graph_text
        if method == "cg":
            assert "HtoD" not in graph_text
    finally:
        if hasattr(inverse, "close"):
            inverse.close()


@pytest.mark.parametrize("selection", ["SA", "LM"])
def test_negative_foundation_native_cupy_diagonal_scaling(selection):
    pytest.importorskip("cupy")
    from benchmarks.applications.references import cupy_setup

    case = make_negative_foundation(4, 6, selection, consistent=False)
    p = case.problem
    solve, close, _ = cupy_setup(p, 48, standard=True, standard_which=selection, eigsh_tol=1e-8)
    try:
        w, v, _ = solve()
        w, v = w.get(), v.get()
        result = accuracy(p, w, v)
        assert result["accepted"], result
        residual = (p.a @ v - (p.b @ v) * w) / np.sqrt(p.b.diagonal())[:, None]
        assert la.norm(residual) <= 1e-8
    finally:
        close()


@pytest.mark.parametrize("selection,consistent", [("SM", False), ("SA", True)])
def test_negative_foundation_native_cupy_unsupported_routes(selection, consistent):
    pytest.importorskip("cupy")
    from benchmarks.applications.references import Unsupported, cupy_setup

    p = make_negative_foundation(4, 6, selection, consistent).problem
    with pytest.raises(Unsupported):
        cupy_setup(p, 48, standard=True, standard_which=selection)
