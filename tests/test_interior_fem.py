"""Interior FEM selection must recover the nearest physical set, not just Ritz pairs."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.applications.interior_fem import NAMES, make_interior_case
from benchmarks.applications.runner import accuracy, warp_output, warp_setup


@pytest.mark.parametrize("name", NAMES)
def test_interior_reference_encloses_requested_band(name):
    c = make_interior_case(name)
    p = c.problem
    la.cholesky(p.b.toarray())
    shifted = la.eigvalsh((p.a - p.target * p.b).toarray())
    assert shifted[0] < 0 < shifted[-1]
    assert c.band_gap[0] < p.target < c.band_gap[1]
    assert c.selection_margin > 0
    assert min(p.expected) < p.target < max(p.expected)
    assert accuracy(p, p.expected, p.requested_basis)["accepted"]
    # Low residual + orthogonality alone would accept these wrong low modes.
    wrong = accuracy(p, c.spectrum[: p.k], c.vectors[:, : p.k])
    assert wrong["residual"] < 1e-8 and wrong["orthogonality"] < 1e-8
    assert not wrong["accepted"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("fraction", [0.37, 0.001])
def test_interior_captured_requested_set(name, fraction):
    c = make_interior_case(name, fraction=fraction)
    p = c.problem
    solver, inverse = warp_setup(p, ncv=48, tol=1e-9)
    graph = solver.capture(12)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            check = accuracy(p, *warp_output(p, solver))
            assert check["accepted"], check
            assert solver.status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == p.k
    finally:
        solver.graph = graph = None
        inverse.close()


def test_distance_boundary_cluster_is_not_split():
    c = make_interior_case("repeated_bending", modes=5)
    assert c.problem.k == 6
    assert np.all(np.bincount(c.selected_indices // 2)[np.unique(c.selected_indices // 2)] == 2)


@pytest.mark.parametrize("fraction", [0.0, 1.0, np.nan])
def test_interior_gap_validation(fraction):
    with pytest.raises(ValueError):
        make_interior_case("clamped_consistent", fraction=fraction)
