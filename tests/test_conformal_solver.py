"""Captured reciprocal solves recover the full original singular pencil."""

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.conformal import boundary_reference, make_conformal_case, metrics
from benchmarks.applications.conformal_warp import ConformalEigensolver
from benchmarks.graph_audit import graph_summary


@pytest.mark.parametrize("height", [0.0, 0.25])
@pytest.mark.parametrize("backend", ["cudss", "cg"])
@pytest.mark.parametrize("pin", [0, 27])
@pytest.mark.parametrize("modes", [2, 6])
def test_changed_starts_preserve_finite_spectrum(height, backend, pin, modes, tmp_path):
    case = make_conformal_case(8, modes, height, pin)
    reference = boundary_reference(case)
    solver = ConformalEigensolver(case, 32, backend)
    graph = solver.capture(8)
    try:
        rng = np.random.default_rng(207)
        for _ in range(2):
            solver.core.initial_vector.assign(rng.normal(size=case.n))
            if backend == "cg":
                solver.inverse.reset_stats()
            wp.capture_launch(graph)
            check = metrics(
                case, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, reference
            )
            assert check["accepted"], check
            assert solver.converged.numpy()[0] == case.k
            assert solver.core.converged.numpy()[0] == case.k
            assert solver.core.status.numpy()[0] == 0
            if backend == "cg":
                assert solver.inverse.stats.numpy()[2] == 0
                assert solver.inverse.fresh_stats.numpy()[2] == 0
        dot = tmp_path / "conformal.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
        assert not any("DtoH" in kind for kind in audit["memcpy_kinds"])
        if backend == "cg":
            assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
    finally:
        graph = None
        solver.close()


def test_invalid_reciprocal_values_cannot_certify_zero_outputs():
    solver = ConformalEigensolver(make_conformal_case(), 32, "cg")
    solver.core.eigenvectors.zero_()
    solver.core.eigenvalues.zero_()
    solver.finalize()
    assert solver.converged.numpy()[0] == 0


def test_short_budget_cannot_certify_missing_repeated_modes():
    case = make_conformal_case(12, 6, 0.25)
    solver = ConformalEigensolver(case, 8)
    graph = solver.capture(1)
    try:
        wp.capture_launch(graph)
        check = metrics(
            case,
            solver.eigenvalues.numpy(),
            solver.eigenvectors.numpy().T,
            boundary_reference(case),
        )
        assert not check["accepted"]
        assert solver.converged.numpy()[0] < case.k
    finally:
        graph = None
        solver.close()
