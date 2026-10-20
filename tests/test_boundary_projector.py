"""Implicit projector products: aliases, batched rows, changing data and capture."""

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.conformal import boundary_reference, make_conformal_case, metrics
from benchmarks.applications.conformal_warp import ConformalEigensolver
from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.boundary_projector import CenteredBoundary, ShiftedConformal


@pytest.mark.parametrize("side,pin", [(8, 0), (8, 27), (70, 0)])
@pytest.mark.parametrize("alias", [False, True])
def test_block_products(side, pin, alias, tmp_path):
    case = make_conformal_case(side, 6, 0.25, pin, shift=-0.001)
    k, m = case.reduced
    rng = np.random.default_rng(72)
    x = wp.empty((3, case.n), dtype=wp.float64)
    y = x if alias else wp.empty_like(x)
    for operator, reference in (
        (CenteredBoundary(case, 3), m),
        (ShiftedConformal(case, 3), k - case.shift * m),
    ):
        x.fill_(1.0)
        operator.apply(x, y)
        wp.synchronize()
        with wp.ScopedCapture() as captured:
            operator.apply(x, y)
        for scale in (1e-6, 1.0, 1e6):
            values = scale * rng.normal(size=x.shape)
            x.assign(values)
            wp.capture_launch(captured.graph)
            np.testing.assert_allclose(
                y.numpy(), (reference @ values.T).T, atol=2e-13 * scale, rtol=2e-13
            )
        dot = tmp_path / "boundary.dot"
        wp.capture_debug_dot_print(captured.graph, str(dot))
        audit = graph_summary(dot)
        assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
        assert all("DtoD" in kind for kind in audit["memcpy_kinds"])


@pytest.mark.parametrize("alias", ["none", "x", "y", "all"])
def test_linear_operator_alpha_beta_aliases(alias):
    case = make_conformal_case(8)
    operator = ShiftedConformal(case, 1)
    k, m = case.reduced
    rng = np.random.default_rng(100)
    a, b = rng.normal(size=(2, case.n))
    x = wp.array(a, dtype=wp.float64)
    y = x if alias == "all" else wp.array(b, dtype=wp.float64)
    if alias == "all":
        b = a
    z = x if alias in ("x", "all") else y if alias == "y" else wp.empty_like(x)
    operator.linear.matvec(x, y, z, 0.7, -0.3)
    expected = 0.7 * ((k - case.shift * m) @ a) - 0.3 * b
    np.testing.assert_allclose(z.numpy(), expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("backend", ["cudss", "cg"])
@pytest.mark.parametrize("height", [0.0, 0.25])
def test_implicit_modes(backend, height):
    case = make_conformal_case(8, 6, height)
    solver = ConformalEigensolver(case, 32, backend, implicit=True)
    reference = boundary_reference(case)
    graph = solver.capture(8)
    try:
        for _ in range(2):
            if backend == "cg":
                solver.inverse.reset_stats()
            wp.capture_launch(graph)
            check = metrics(
                case, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, reference
            )
            assert check["accepted"], check
            assert solver.converged.numpy()[0] == case.k
            assert solver.core.converged.numpy()[0] == case.k
            if backend == "cg":
                assert solver.inverse.fresh_stats.numpy()[2] == 0
    finally:
        graph = None
        solver.close()
