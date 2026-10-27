"""Two-pass inverse application must preserve mass orthogonality and cached AX."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.experimental.lower_triangular_rotate import LowerTriangularRotateBlock
from benchmarks.experimental.triangular_rotate import TriangularRotateBlock
from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator


class AllSizes(TriangularRotateBlock):
    _inverse_minimum = 2


class AllSizesLower(LowerTriangularRotateBlock):
    _inverse_minimum = 2


@pytest.mark.parametrize("candidate", [AllSizes, AllSizesLower])
@pytest.mark.parametrize("width", [8, 52, 104, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_orthogonalization_with_changed_mass_scaling_and_cached_images(width, cached, candidate):
    n = 3 * width + 7
    rng = np.random.default_rng(173)
    source = rng.normal(size=(width, n))
    diagonal = np.linspace(1, 4, n)
    a = sp.diags(diagonal, format="csr")
    mass_host = sp.diags(
        [-0.1 * np.ones(n - 1), np.linspace(1, 2, n), -0.1 * np.ones(n - 1)],
        [-1, 0, 1],
        format="csr",
    )
    mass = matrix(mass_host)
    solvers = []
    graphs = []
    for constructor in [BlockLOBPCG, candidate]:
        solver = constructor(
            SparseOperator(matrix(a)),
            4,
            width,
            mass=SparseOperator(mass),
            reuse_operator_images=cached,
        )
        solver._image_reuse_active = cached
        solver.sx.assign(source)
        if cached:
            solver._image_x.assign(source * diagonal)
        solver.orthogonalize_x()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            solver.status.zero_()
            solver.orthogonalize_x()
        # Complete lazy graph upload at setup, before the changed-input H2D
        # copies below. Driver 570 can otherwise let the first graph observe a
        # partial upload (#61/#198); the eigensolver-free diagnostic is in
        # benchmarks/experimental/graph_input_upload.py. This does not add a
        # host barrier between the numerical operations or subsequent replays.
        wp.capture_launch(capture.graph)
        wp.synchronize()
        solvers.append(solver)
        graphs.append(capture.graph)
    for kind in ["normal", "scaled", "correlated", "changed_mass", "equilibrated_mass", "normal"]:
        current = source.copy()
        m = mass_host if kind != "changed_mass" else 1.25 * mass_host
        if kind == "scaled":
            current *= np.geomspace(1e-60, 1e60, width)[:, None]
        elif kind == "correlated":
            current[1] = current[0] + 1e-3 * current[1]
        elif kind == "equilibrated_mass":
            d = np.geomspace(1e-6, 1e6, n)
            m = (sp.diags(d) @ mass_host @ sp.diags(d)).tocsr()
            m.sort_indices()
            current /= d[None, :]
        mass.values.assign(m.data)
        outputs = []
        for solver, graph in zip(solvers, graphs):
            solver.sx.assign(current)
            if cached:
                solver._image_x.assign(current * diagonal)
            wp.capture_launch(graph)
            assert solver.status.numpy()[0] == 0
            x = solver.sx.numpy()
            assert la.norm(x @ (m @ x.T) - np.eye(width)) < 1e-10
            np.testing.assert_allclose(solver.bx.numpy(), (m @ x.T).T, atol=1e-13, rtol=1e-12)
            if cached:
                image = solver._image_x.numpy()
                assert la.norm(image - x * diagonal) / la.norm(image) < 1e-10
            outputs.append(x)
        # Equivalent complete search space, allowing different roundoff/basis.
        overlap = outputs[0] @ (m @ outputs[1].T)
        assert la.norm(overlap @ overlap.T - np.eye(width)) < 1e-9
