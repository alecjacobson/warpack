"""Relative-only CG inverse comparisons across units and SPD mass structures."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.compare import compare_baseline
from benchmarks.applications.negative_foundation import make_negative_foundation
from benchmarks.applications.runner import accuracy, warp_setup


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("selection", ["SA", "LM"])
def test_relative_cg_scaled_generalized_pencil(consistent, selection):
    original = make_negative_foundation(3, 6, selection, consistent).problem
    for scale in (1.0, 1e-8, 1e8):
        p = replace(
            original,
            a=original.a * scale,
            b=original.b * scale,
            requested_basis=original.requested_basis / np.sqrt(scale),
        )
        solver, inverse = warp_setup(p, method="cg", ncv=p.n, tol=1e-10, cg_tol=1e-12, cg_atol=0.0)
        graph = solver.capture(3)
        inverse.reset_stats()
        for _ in range(2):
            wp.capture_launch(graph)
            result = accuracy(p, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
            assert result["accepted"], (scale, result)
            assert solver.status.numpy()[0] == 0
            assert solver.converged.numpy()[0] == p.k
        assert inverse.stats.numpy()[2] == 0


def test_cg_absolute_tolerance_is_part_of_regression_identity():
    old = dict(
        api_class="C",
        n=1024,
        method="warp-cg",
        matrix_sha256="same-pencil",
        accepted=True,
        timing_uncontended=True,
        solve_seconds=1.0,
        cg_tolerance=1e-13,
    )
    default = compare_baseline([{**old, "cg_absolute_tolerance": None}], {"cases": [old]}, 1.25)[0]
    assert default["comparable"] and not default["performance_regression"]
    relative = compare_baseline([{**old, "cg_absolute_tolerance": 0.0}], {"cases": [old]}, 1.25)[0]
    assert not relative["comparable"] and "performance_regression" not in relative

    legacy_iterations = compare_baseline(
        [{**old, "reference_maxiter": 200}], {"cases": [old]}, 1.25
    )[0]
    assert legacy_iterations["comparable"]
    changed_iterations = compare_baseline(
        [{**old, "reference_maxiter": 1000}], {"cases": [old]}, 1.25
    )[0]
    assert not changed_iterations["comparable"]
