"""The public lowest-mode block solver also targets dominant modes through -A."""

import numpy as np
import pytest
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import audit
from benchmarks.applications.runner import accuracy
from benchmarks.applications.spectral_surface import surface_pencil
from benchmarks.applications.spectral_surface_compare import problem
from benchmarks.experimental.spectral_surface_block import SurfaceBlockSolver


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("gamma", [0.0, 0.001])
def test_dominant_block_complete_cluster_and_changed_replay(copies, gamma, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = problem(surface_pencil(4, gamma=gamma), copies, copies)
    driver = SurfaceBlockSolver(p)
    driver.capture()
    for eta in (10, 30):
        current = problem(surface_pencil(4, gamma=gamma, anisotropy=eta), copies, copies)
        np.testing.assert_array_equal(current.a.indices, p.a.indices)
        driver.a.values[: current.a.nnz].assign(-current.a.data)
        driver.solve()
        q = accuracy(current, *driver.output())
        assert q["accepted"], q
        assert q["residual"] < 1e-11
        assert driver.solver.status.numpy()[0] == 0
        assert driver.solver.converged.numpy()[0] == copies
    audit(driver.graph, tmp_path / "surface-block.dot")
