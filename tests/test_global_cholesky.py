"""Independent factor equations and replay for the global-workspace prototype."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.experimental import profile_active_block
from benchmarks.experimental.global_cholesky import GlobalCholeskyBlock, global_cholesky
from warpack import kernels as K


@pytest.mark.parametrize("n", [1, 7, 48, 104, 128, 204, 256, 512])
@pytest.mark.parametrize("scaled", [False, True])
def test_global_factor_strides_replay_and_original_equation(n, scaled):
    rng = np.random.default_rng(155)
    a = rng.normal(size=(n, n))
    original = a @ a.T + np.eye(n)
    if scaled:
        d = np.geomspace(1, 1e-5, n)
        original *= d[:, None] * d[None, :]
    source_storage = wp.empty((n, n + 3), dtype=wp.float64)
    source = source_storage[:, :n]
    output_storage = wp.full((n, n + 3), 17.0, dtype=wp.float64)
    output = output_storage[:, :n]
    work_storage = wp.full((n, n + 3), np.nan, dtype=wp.float64)
    work = work_storage[:, :n]
    serial = wp.zeros((n, n), dtype=wp.float64)
    flag, serial_flag = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
    with wp.ScopedCapture() as capture:
        flag.zero_()
        wp.launch_tiled(
            global_cholesky,
            dim=1,
            inputs=[source, output, flag, work],
            block_dim=256 if n <= 256 else 512,
        )
    for scale in (1.0, 3.0, 0.5):
        current = original * scale
        source.assign(current)
        wp.capture_launch(capture.graph)
        wp.launch(K.cholesky, 1, [source, serial, serial_flag])
        assert flag.numpy()[0] == serial_flag.numpy()[0] == 0
        factor = np.tril(output.numpy())
        np.testing.assert_allclose(factor, la.cholesky(current, lower=True), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(factor, np.tril(serial.numpy()), rtol=1e-12, atol=1e-13)
        np.testing.assert_array_equal(np.triu(work.numpy()), factor.T)
        assert np.linalg.norm(factor @ factor.T - current) / np.linalg.norm(current) < 1e-12
        assert np.all(output_storage.numpy()[:, n:] == 17.0)
        assert np.all(output.numpy()[np.triu_indices(n, 1)] == 17.0)
        assert np.all(np.isnan(work_storage.numpy()[:, n:]))
        assert np.all(np.isnan(work.numpy()[np.tril_indices(n, -1)]))


@pytest.mark.parametrize("pivot", [0, 17])
@pytest.mark.parametrize("value", [0.0, -1e-40, np.nan, np.inf])
def test_global_factor_invalid_pivot_sticky_status_and_recovery(pivot, value):
    n = 24
    source = wp.array(np.eye(n), dtype=wp.float64)
    output, work, serial = (wp.empty_like(source) for _ in range(3))
    flag, serial_flag = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
    with wp.ScopedCapture() as capture:
        wp.launch_tiled(global_cholesky, dim=1, inputs=[source, output, flag, work], block_dim=256)
        wp.launch(K.cholesky, 1, [source, serial, serial_flag])
    bad = np.eye(n)
    bad[pivot, pivot] = value
    for current, reset, expected in ((bad, False, 1), (np.eye(n), False, 1), (np.eye(n), True, 0)):
        source.assign(current)
        if reset:
            flag.zero_()
            serial_flag.zero_()
        wp.capture_launch(capture.graph)
        assert flag.numpy()[0] == serial_flag.numpy()[0] == expected
        np.testing.assert_allclose(np.tril(output.numpy()), np.tril(serial.numpy()), equal_nan=True)
        if np.array_equal(current, np.eye(n)):
            np.testing.assert_array_equal(np.tril(output.numpy()), current)


@pytest.mark.parametrize("case", ["clamped_consistent", "free", "heterogeneous"])
def test_global_factor_workspace_shared_with_active_branches(case, monkeypatch):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(case=case, side=3, modes=6, width=48)
    monkeypatch.setattr(profile_active_block, "BlockLOBPCG", GlobalCholeskyBlock)
    solver, check, _, _ = profile_active_block.create_solver(SimpleNamespace(**config))
    assert all(
        branch._factor_workspace is solver._factor_workspace for branch in solver._search.buckets
    )
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        result = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert result["accepted"], result
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == solver.k
