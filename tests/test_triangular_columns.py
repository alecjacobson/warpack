"""Column grouping preserves equations, partial tiles and full physical spaces."""

from functools import partial

import pytest

from benchmarks.experimental.blocked_triangular_columns import ColumnTriangularBlock, column_kernel
from tests import test_blocked_triangular as original


@pytest.mark.parametrize("columns", [32, 64, 128])
@pytest.mark.parametrize("width", [7, 17, 104, 120, 204])
def test_column_equations_and_replay(monkeypatch, columns, width):
    monkeypatch.setattr(original, "blocked_triangular", column_kernel(columns))
    original.test_equations_strides_and_replay(width, columns)


@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("width", [120, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_column_full_stage(columns, width, cached):
    original.check_full_stage(width, cached, partial(ColumnTriangularBlock, column_tile=columns))


@pytest.mark.parametrize("columns", [64, 128])
def test_poisoned_upper_triangle(monkeypatch, columns):
    monkeypatch.setattr(original, "blocked_triangular", column_kernel(columns))
    original.test_unused_upper_factor_triangle_is_never_read(120, columns)
