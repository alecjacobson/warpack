"""Cooperative factorization preserves triangular equations, failures and replay."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.experimental import profile_active_block
from benchmarks.experimental.parallel_cholesky import (
    CooperativeCholeskyBlock,
    packed_cholesky_kernel,
)
from warpack import kernels as K


@pytest.mark.parametrize("n", [1, 3, 7, 24, 64, 104, 128])
@pytest.mark.parametrize("scaled", [False, True])
def test_packed_cholesky_factor_and_changed_strided_input(n, scaled):
    if wp.get_device().max_shared_memory_per_block < 8 * (n * (n + 1) // 2 + 1):
        pytest.skip("insufficient shared memory for the experimental factor")
    rng = np.random.default_rng(1543)
    a = rng.normal(size=(n, n))
    g = a @ a.T + np.eye(n)
    if scaled:
        d = np.geomspace(1, 1e-5, n)
        g *= d[:, None] * d[None, :]
    storage = wp.zeros((n, n + 3), dtype=wp.float64)
    source = storage[:, :n]
    out_storage = wp.full((n, n + 3), 17.0, dtype=wp.float64)
    result = out_storage[:, :n]
    serial = wp.zeros((n, n), dtype=wp.float64)
    flag, serial_flag = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
    kernel = packed_cholesky_kernel(n)
    with wp.ScopedCapture() as capture:
        flag.zero_()
        wp.launch_tiled(kernel, dim=1, inputs=[source, result, flag], block_dim=128)
    for scale in (1.0, 3.0, 0.5):
        current = g * scale
        source.assign(current)
        wp.capture_launch(capture.graph)
        wp.launch(K.cholesky, 1, [source, serial, serial_flag])
        assert flag.numpy()[0] == serial_flag.numpy()[0] == 0
        factor = np.tril(result.numpy())
        expected = la.cholesky(current, lower=True)
        np.testing.assert_allclose(factor, expected, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(factor, np.tril(serial.numpy()), rtol=1e-12, atol=1e-14)
        assert np.linalg.norm(factor @ factor.T - current) / np.linalg.norm(current) < 1e-12
        assert np.all(out_storage.numpy()[:, n:] == 17.0)


@pytest.mark.parametrize("bad", ["indefinite", "zero", "nan", "inf"])
def test_packed_cholesky_status_and_recovery(bad):
    n = 24
    source = wp.array(np.eye(n), dtype=wp.float64)
    result, serial = wp.empty_like(source), wp.empty_like(source)
    flag, serial_flag = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
    with wp.ScopedCapture() as capture:
        flag.zero_()
        serial_flag.zero_()
        wp.launch_tiled(
            packed_cholesky_kernel(n), dim=1, inputs=[source, result, flag], block_dim=128
        )
        wp.launch(K.cholesky, 1, [source, serial, serial_flag])
    invalid = np.eye(n)
    invalid[0, 0] = {"indefinite": -1, "zero": 0, "nan": np.nan, "inf": np.inf}[bad]
    for current, expected in ((invalid, 1), (np.eye(n), 0)):
        source.assign(current)
        wp.capture_launch(capture.graph)
        assert flag.numpy()[0] == serial_flag.numpy()[0] == expected
        if expected == 0:
            np.testing.assert_array_equal(np.tril(result.numpy()), np.eye(n))


@pytest.mark.parametrize("case", ["clamped_consistent", "free", "heterogeneous"])
def test_cooperative_factor_in_active_fem_replay(case, monkeypatch):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(case=case, side=2, modes=6, width=16)
    monkeypatch.setattr(profile_active_block, "BlockLOBPCG", CooperativeCholeskyBlock)
    solver, check, _, _ = profile_active_block.create_solver(SimpleNamespace(**config))
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        quality = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert solver.status.numpy()[0] == 0
        assert solver.converged.numpy()[0] == solver.k
        assert quality["accepted"], quality


@pytest.mark.parametrize("pivot", [1e-40, -1e-40, np.nan, np.inf])
def test_late_pivot_preserves_serial_containment_and_sticky_status(pivot):
    # A late failing pivot exercises synchronization across earlier columns.
    # Tiny positive pivots intentionally keep the old 1e-30 containment floor;
    # this is a compatibility check, not a claim of an accurate SPD factor.
    n = 24
    original = np.eye(n)
    original[17, 17] = pivot
    source = wp.array(original, dtype=wp.float64)
    result = wp.full((n, n), 17.0, dtype=wp.float64)
    serial = wp.full((n, n), 17.0, dtype=wp.float64)
    flag, serial_flag = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
    with wp.ScopedCapture() as capture:
        wp.launch_tiled(
            packed_cholesky_kernel(n), dim=1, inputs=[source, result, flag], block_dim=128
        )
        wp.launch(K.cholesky, 1, [source, serial, serial_flag])
    wp.capture_launch(capture.graph)
    assert flag.numpy()[0] == serial_flag.numpy()[0] == int(not (0 < pivot < np.inf))
    np.testing.assert_allclose(result.numpy(), serial.numpy(), rtol=1e-14, equal_nan=True)
    assert np.all(result.numpy()[np.triu_indices(n, 1)] == 17.0)
    # A factorization must not erase errors from preceding solver operations.
    source.assign(np.eye(n))
    flag.fill_(1)
    serial_flag.fill_(1)
    wp.capture_launch(capture.graph)
    assert flag.numpy()[0] == serial_flag.numpy()[0] == 1
    np.testing.assert_array_equal(np.tril(result.numpy()), np.eye(n))
