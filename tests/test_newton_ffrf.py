"""Generated FFRF pencils: mechanics, mode completeness, and captured replay."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
import warp.optim.linear as linear

from benchmarks.applications.fem_cases import rigid_modes
from benchmarks.applications.newton_ffrf import (
    EIGENVALUE_SCALE,
    NODE_MASS,
    SPACING,
    SPRING,
    check_pairs,
    gpu_solver,
    lattice,
    upstream,
)
from benchmarks.graph_audit import graph_summary


@pytest.mark.parametrize("boundary", ["free", "fixed"])
def test_lattice_mechanics(boundary):
    k, m, points, retained = lattice((6, 4, 3), boundary)
    scaled = k.toarray() / (SPRING / SPACING)
    np.testing.assert_allclose(scaled, scaled.T, atol=1e-14)
    spectrum = la.eigvalsh(scaled)
    assert spectrum.min() > -1e-13
    assert np.count_nonzero(abs(spectrum) < 1e-10) == (6 if boundary == "free" else 0)
    np.testing.assert_array_equal(m.diagonal(), np.full(len(retained), NODE_MASS))
    if boundary == "free":
        rigid = rigid_modes(points, m)
        assert np.max(abs(scaled @ rigid)) < 1e-12
        # An affine axial stretch has strictly positive spring energy.
        stretch = np.zeros_like(points)
        stretch[:, 0] = points[:, 0]
        assert stretch.ravel() @ scaled @ stretch.ravel() > 0
    else:
        global_nodes = retained // 3
        assert np.all(points[global_nodes, 0] > points[:, 0].min())
        assert np.all(points[global_nodes, 0] < points[:, 0].max())


@pytest.mark.parametrize("backend", ["cudss", "warp"])
@pytest.mark.parametrize("boundary", ["free", "fixed"])
def test_captured_modes_match_dense_and_reject_wrong_spectrum(tmp_path, backend, boundary):
    if backend == "warp" and not hasattr(linear, "FSAI"):
        pytest.skip("requires the documented upstream Warp FSAI overlay")
    k, m, points, _ = lattice((8, 5, 3), boundary)
    normalized, vectors = la.eigh(k.toarray() / (SPRING / SPACING))
    first = 6 if boundary == "free" else 0
    count = 12
    reference = (
        normalized[first : first + count] * EIGENVALUE_SCALE,
        vectors[:, first : first + count] / np.sqrt(NODE_MASS),
    )
    assert normalized[first + count] / normalized[first + count - 1] > 1 + 1e-6
    rigid = rigid_modes(points, m) if boundary == "free" else None
    sigma = -1e-6 * float(abs(k).sum(axis=0).max()) / NODE_MASS if rigid is not None else 0
    solver, graph, values, modes, inverse, _ = gpu_solver(
        k, points, boundary, backend, count, sigma
    )
    try:
        dot = tmp_path / "graph.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        assert not graph_summary(dot)["node_types_including_conditional_body"].get("HOST", 0)
        for _ in range(2):
            wp.capture_launch(graph)
            assert solver.status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == count
            assert check_pairs(k, m, values.numpy(), modes.numpy().T, reference, rigid)["accepted"]
        # Exact eigenpairs can still be the wrong requested spectrum. Exercise
        # the benchmark's coverage gate, not just its equation-residual gate.
        wrong = slice(first + 1, first + count + 1)
        bad = check_pairs(
            k,
            m,
            normalized[wrong] * EIGENVALUE_SCALE,
            vectors[:, wrong] / np.sqrt(NODE_MASS),
            reference,
            rigid,
        )
        assert bad["relative_residual"] < 1e-10
        assert not bad["accepted"]
    finally:
        wp.synchronize()
        solver.graph = None
        del graph, solver
        if inverse is not None:
            inverse.close()


def test_upstream_reference_requires_pinned_source(tmp_path):
    path = tmp_path / "modal_sparse.py"
    path.write_text("raise RuntimeError('must not execute unverified source')")
    with pytest.raises(ValueError, match="pinned commit"):
        upstream(path)
