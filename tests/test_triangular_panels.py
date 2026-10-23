"""Panel width must not compromise triangular equations or complete QR passes."""

from functools import partial

import pytest

from benchmarks.experimental.blocked_triangular_panels import PanelTriangularBlock, panel_kernel
from tests import test_blocked_triangular as original


@pytest.mark.parametrize("panel_rows", [16, 32])
@pytest.mark.parametrize("width", [7, 17, 52, 104, 204])
def test_panel_equations_and_replay(monkeypatch, panel_rows, width):
    monkeypatch.setattr(original, "blocked_triangular", panel_kernel(panel_rows))
    original.test_equations_strides_and_replay(width)


@pytest.mark.parametrize("panel_rows", [16, 32])
@pytest.mark.parametrize("width", [52, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_panel_full_stage(panel_rows, width, cached):
    original.check_full_stage(width, cached, partial(PanelTriangularBlock, panel_rows=panel_rows))
