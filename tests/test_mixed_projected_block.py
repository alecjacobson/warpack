"""FP32 projected starts must retain complete original FEM solve acceptance."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import warp as wp
from warp.optim import linear

from benchmarks.experimental import profile_active_block
from benchmarks.experimental.compare_mixed_projected import projected_factory
from warpack import projected


@pytest.mark.parametrize("limit", [64, 72])
@pytest.mark.parametrize("case", ["clamped_consistent", "free", "heterogeneous"])
def test_complete_fem_replay_with_mixed_projected_start(case, limit, monkeypatch):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    if wp.get_device().arch != 89:
        pytest.skip("candidate guard targets the measured Ada global path")
    dimensions = []
    monkeypatch.setattr(
        projected, "GlobalJacobi", projected_factory(projected.GlobalJacobi, limit, dimensions)
    )
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(case=case, side=3, modes=6, width=24)
    solver, check, _, _ = profile_active_block.create_solver(SimpleNamespace(**config))
    assert dimensions and all(48 <= n <= limit for n in dimensions)
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        quality = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert quality["accepted"], quality
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == solver.k
