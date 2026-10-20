"""Exact upstream diagonal mass inversion keeps the generalized metric."""

from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.applications.negative_foundation import make_negative_foundation
from benchmarks.applications.runner import accuracy, has_positive_diagonal_mass, warp_setup


@pytest.mark.parametrize("method", ["diagonal", "scaled"])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_regular_exact_diagonal_mass(scale, method, tmp_path):
    p = make_negative_foundation(3, 6, "LM", consistent=False).problem
    # A simultaneous nonuniform coordinate congruence preserves eigenvalues
    # but requires componentwise recovery and the original mass metric.
    coordinates = np.linspace(0.2, 5.0, p.n)
    c = sp.diags(coordinates, format="csr")
    p = replace(
        p,
        a=(c @ p.a @ c) * scale,
        b=(c @ p.b @ c) * scale,
        requested_basis=p.requested_basis / (coordinates[:, None] * np.sqrt(scale)),
    )
    solver, inverse = warp_setup(p, method=method, ncv=p.n)
    if inverse is not None:
        assert not inverse.requires_static_outer_capture
    graph = solver.capture(3)
    for _ in range(2):
        wp.capture_launch(graph)
        q = accuracy(p, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert q["accepted"], q
        assert solver.converged.numpy()[0] == p.k
        assert solver.status.numpy()[0] == 0
    dot = tmp_path / "regular-diagonal.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(word in dot.read_text() for word in ("HOST", "HtoD", "DtoH"))


def test_regular_diagonal_rejects_approximate_or_invalid_inverse():
    p = make_negative_foundation(3, 6, "LM", consistent=True).problem
    assert not has_positive_diagonal_mass(p)
    p = replace(p, b=sp.eye(p.n, format="csr"))
    assert has_positive_diagonal_mass(p)
    assert not has_positive_diagonal_mass(replace(p, mode="shift-invert"))
    assert not has_positive_diagonal_mass(replace(p, b=None))
    for entry in [0, -1, np.nan, np.inf]:
        diagonal = np.ones(p.n)
        diagonal[0] = entry
        assert not has_positive_diagonal_mass(replace(p, b=sp.diags(diagonal, format="csr")))
