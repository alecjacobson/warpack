"""Register-blocked recurrence in complete constrained and clustered FEM solves."""

import pytest

from benchmarks.experimental.register_triangular import RegisterTriangularBlock
from tests import test_blocked_fem as original


@pytest.mark.parametrize(
    "case", ["clamped_consistent", "clamped_lumped", "free", "disconnected", "heterogeneous"]
)
@pytest.mark.parametrize("keep_history", [False, True])
def test_complete_fem(monkeypatch, case, keep_history):
    monkeypatch.setattr(original, "BlockedTriangularBlock", RegisterTriangularBlock)
    original.test_complete_fem(case, keep_history)


@pytest.mark.parametrize("keep_history", [False, True])
def test_changed_coupled_pencil(monkeypatch, tmp_path, keep_history):
    monkeypatch.setattr(original, "BlockedTriangularBlock", RegisterTriangularBlock)
    original.test_changed_mass_constraints_clusters(monkeypatch, tmp_path, keep_history)
