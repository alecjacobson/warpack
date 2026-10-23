"""Fresh images must preserve complete QR, physical modes and changed-input replay."""

import pytest

from benchmarks.experimental.ritz_image_refresh import FreshRitzImageBlock
from tests import test_certified_whitening, test_first_whitening
from tests.test_triangular_rotate import (
    test_orthogonalization_with_changed_mass_scaling_and_cached_images as check_stage,
)


@pytest.mark.parametrize("width", [24, 96, 104, 120, 204])
@pytest.mark.parametrize("cached", [False, True])
def test_full_stage(width, cached):
    check_stage(width, cached, FreshRitzImageBlock)


@pytest.mark.parametrize("history", [False, True])
def test_changed_coupled_constraints_clusters(monkeypatch, tmp_path, history):
    monkeypatch.setattr(test_first_whitening, "AllSizeInstalledWhitening", FreshRitzImageBlock)
    test_certified_whitening.test_coupled_mass_constraints_clusters_and_changed_problem(
        tmp_path, "installed", history, problem_shape=(336, 20, 104)
    )
