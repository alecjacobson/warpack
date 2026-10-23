"""Complete FEM and changed coupled pencils with direct blocked substitution."""

import pytest

from benchmarks.experimental.blocked_triangular import BlockedTriangularBlock
from tests import test_certified_whitening, test_first_whitening


@pytest.mark.parametrize(
    "case", ["clamped_consistent", "clamped_lumped", "free", "disconnected", "heterogeneous"]
)
@pytest.mark.parametrize("keep_history", [True, False])
def test_complete_fem(case, keep_history):
    test_first_whitening.test_fem_modes_and_nullspace_with_first_certificate(
        case, keep_history, BlockedTriangularBlock
    )


@pytest.mark.parametrize("keep_history", [True, False])
def test_changed_mass_constraints_clusters(monkeypatch, tmp_path, keep_history):
    class AllSizeBoth(BlockedTriangularBlock):
        _first_whitening_minimum = 2
        _first_whitening_arches = None

    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", AllSizeBoth)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", keep_history
    )
