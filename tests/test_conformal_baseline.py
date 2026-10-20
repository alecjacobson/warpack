"""Changing a tolerance must invalidate a timing baseline."""

from copy import deepcopy

import pytest
from test_connection_baseline import report

from benchmarks.applications.conformal_compare import assess_baseline


@pytest.mark.parametrize("field", ["outer_tol", "cupy_tol", "cupy_maxiter", "reciprocal_scale"])
def test_changed_conformal_contract_is_incomparable(field):
    old = report()
    old["cases"][0].update(outer_tol=1e-10, cupy_tol=1e-9, cupy_maxiter=300, reciprocal_scale=1.0)
    current = deepcopy(old)
    assess_baseline(current, old)
    assert current["baseline_comparable"]
    current["cases"][0][field] *= 2
    assess_baseline(current, old)
    assert not current["baseline_comparable"]
    assert "time_ratio" not in current["cases"][0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("inner_precision", "float64"),
        ("inner_steps", 10),
        ("block_width", 20),
        ("outer_iteration_budget", 100),
    ],
)
@pytest.mark.parametrize("method", ["warp-block-cg", "cupy-cg"])
def test_changed_block_contract_is_incomparable(field, value, method):
    old = report()
    old["cases"][0].update(
        method=method,
        inner_precision="float32",
        inner_steps=25,
        block_width=24,
        outer_iteration_budget=200,
    )
    current = deepcopy(old)
    assess_baseline(current, old)
    assert current["baseline_comparable"]
    current["cases"][0][field] = value
    assess_baseline(current, old)
    assert not current["baseline_comparable"]
