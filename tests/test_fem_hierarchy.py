"""Independent finite-element identities and captured computed-basis refinement."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import (
    BeamTransfer,
    attach_reference,
    beam_problem,
    transfer_stencil,
)
from benchmarks.applications.fem_shapes import ShapeReplay


def host_transfer(coarse, fine):
    ids, weights = transfer_stencil(coarse, fine)
    rows = np.broadcast_to(np.arange(len(ids))[:, None], ids.shape)
    keep = ids >= 0
    return sp.csr_matrix(
        (weights[keep], (rows[keep], ids[keep])), shape=(len(ids), 9 * coarse * (coarse + 1) ** 2)
    )


@pytest.mark.parametrize("coarse,fine", [(1, 2), (2, 4), (1, 3), (2, 2)])
def test_nested_p1_galerkin_and_affine_reproduction(coarse, fine):
    a, b = beam_problem(coarse, 4), beam_problem(fine, 4)
    p = host_transfer(coarse, fine)
    # Independent strain/mass assembly fixes interpolation, ordering and boundaries.
    np.testing.assert_allclose((p.T @ b.a @ p).toarray(), a.a.toarray(), atol=3e-13)
    np.testing.assert_allclose((p.T @ b.b @ p).toarray(), a.b.toarray(), atol=3e-15)

    def field(side):
        grid = np.array(list(np.ndindex((3 * side + 1, side + 1, side + 1))), float) / side
        return (grid[grid[:, 0] > 0, :1] * np.array([0.2, -1.3, 2.1])).ravel()

    np.testing.assert_allclose(p @ field(coarse), field(fine), atol=2e-15)
    ids, weights = transfer_stencil(coarse, fine)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert ids.max() < a.n


@pytest.mark.parametrize("coarse,fine", [(0, 2), (3, 2), (2, 3)])
def test_reject_nonnested_grid(coarse, fine):
    with pytest.raises(ValueError, match="integer multiple"):
        transfer_stencil(coarse, fine)


@pytest.mark.parametrize("consistent", [False, True])
def test_computed_coarse_modes_refine_and_replay(consistent, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    coarse_p, fine_p = [beam_problem(s, 6, consistent) for s in (2, 4)]
    for p in (coarse_p, fine_p):
        attach_reference(p)
    coarse = ShapeReplay([(0, coarse_p, 0)], 12, refit=False)
    fine = ShapeReplay([(0, fine_p, 0)], 12, refit=False)
    coarse.solve()
    assert coarse.quality(coarse_p)["accepted"]
    transfer = BeamTransfer(2, 4, coarse.solver.eigenvectors, fine.starts)
    graph = fine.solver.capture(200, initializer=transfer.apply)
    original = coarse.solver.eigenvectors.numpy()
    for sign in (1.0, -1.0):
        coarse.solver.eigenvectors.assign(sign * original)
        wp.capture_launch(graph)
        assert fine.quality(fine_p)["accepted"]
        # Starts remain the raw P1 field; the solver normalizes its internal basis.
        np.testing.assert_allclose(
            fine.starts.numpy(), (host_transfer(2, 4) @ (sign * original).T).T, atol=2e-14
        )
    dot = tmp_path / "hierarchy.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(x in dot.read_text() for x in ("HOST", "HtoD", "DtoH"))
