"""Inexact coarse modes must refine to strict fine modes without reference seeds."""

import numpy as np
import pytest
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import BeamTransfer, attach_reference, beam_problem
from benchmarks.applications.fem_hierarchy_tuning import source_usable
from benchmarks.applications.fem_shapes import ShapeReplay


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_loose_coarse_basis_reaches_strict_requested_fine_space(consistent, full, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    k, width = 6, 12
    count = width if full else k
    coarse_p, fine_p = beam_problem(2, count, consistent), beam_problem(4, k, consistent)
    for p in (coarse_p, fine_p):
        attach_reference(p)
    coarse = ShapeReplay([(0, coarse_p, 0)], count + width - k, refit=False, tol=1e-3)
    fine = ShapeReplay([(0, fine_p, 0)], width, refit=False, steps=None, start_rows=count)
    assert fine.graph is None
    with pytest.raises(ValueError, match="capture"):
        fine.solve()
    transfer = BeamTransfer(2, 4, coarse.solver.eigenvectors, fine.starts)
    fine.graph = fine.solver.capture(200, initializer=transfer.apply)
    for _ in range(2):
        coarse.solve()
        cq = coarse.quality(coarse_p)
        assert source_usable(cq, count, 1e-3), cq
        assert cq["residual"] > 1e-8  # Deliberately not a strict source eigenbasis.
        wp.capture_launch(fine.graph)
        fq = fine.quality(fine_p)
        assert fq["accepted"], fq
        assert fq["residual"] < 1e-11
        np.testing.assert_allclose(fine.solver.eigenvalues.numpy(), fine_p.expected, rtol=1e-9)
    if full:
        with pytest.raises(ValueError, match="one start row"):
            fine.solve(warm=True)
    dot = tmp_path / "loose-transfer.dot"
    wp.capture_debug_dot_print(fine.graph, str(dot))
    assert not any(s in dot.read_text() for s in ("HOST", "DtoH", "HtoD"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", 1),
        ("converged", 3),
        ("residual", 0.01),
        ("orthogonality", float("nan")),
        ("spectrum_error", float("inf")),
    ],
)
def test_loose_source_cannot_hide_failure(field, value):
    q = dict(
        status=0,
        converged=6,
        residual=1e-4,
        orthogonality=1e-14,
        spectrum_error=1e-3,
        requested_subspace_error=1e-2,
    )
    assert source_usable(q, 6, 1e-3)
    q[field] = value
    assert not source_usable(q, 6, 1e-3)
