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
    SerialCholeskyBlock,
)
from warpack import BlockLOBPCG
from warpack import kernels as K
from warpack._projected_cholesky import packed_cholesky_kernel


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
@pytest.mark.parametrize("backend", ["prototype", "default"])
def test_cooperative_factor_in_active_fem_replay(case, backend, monkeypatch):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    config = json.loads(Path("results/cg_partitioned/final/exact-a1.json").read_text())["args"]
    config.update(case=case, side=3, modes=6, width=48)
    monkeypatch.setattr(
        profile_active_block,
        "BlockLOBPCG",
        CooperativeCholeskyBlock if backend == "prototype" else BlockLOBPCG,
    )
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


@pytest.mark.parametrize(
    "size,cuda,shared,expected",
    [
        (4, True, 100000, "serial"),
        (24, True, 100000, "serial"),
        (47, True, 100000, "serial"),
        (48, True, 100000, "cooperative"),
        (104, True, 49152, "cooperative"),
        (128, True, 100000, "cooperative"),
        (128, True, 49152, "serial"),
        (129, True, 100000, "global"),
        (204, True, 49152, "global"),
        (256, True, 49152, "global"),
        (512, True, 49152, "global"),
        (513, True, 100000, "serial"),
        (204, False, None, "serial"),
        (104, True, 44711, "serial"),
        (104, True, 44712, "cooperative"),
        (104, False, None, "serial"),
    ],
)
def test_production_factor_dispatch_and_capacity_fallback(
    size, cuda, shared, expected, monkeypatch
):
    # Static launch metadata can be tested without allocating a large eigensolver
    # or requiring the test GPU to have the simulated capacity.
    device = SimpleNamespace(is_cuda=cuda)
    if shared is not None:
        device.max_shared_memory_per_block = shared
    work = object() if cuda and 129 <= size <= 512 else None
    owner = SimpleNamespace(device=device, _global_factor_workspace=work)
    arguments = [SimpleNamespace(shape=(size, size)), object(), object()]
    calls = []
    monkeypatch.setattr(wp, "launch", lambda *a, **kw: calls.append(("serial", a, kw)))
    monkeypatch.setattr(
        wp,
        "launch_tiled",
        lambda *a, **kw: calls.append(("global" if work is not None else "cooperative", a, kw)),
    )
    BlockLOBPCG.launch(owner, K.cholesky, 1, arguments)
    assert len(calls) == 1 and calls[0][0] == expected
    assert calls[0][2]["device"] is device
    if expected == "cooperative":
        assert calls[0][1][0] is packed_cholesky_kernel(size)
        assert calls[0][2]["inputs"] is arguments
        assert calls[0][2]["dim"] == 1 and calls[0][2]["block_dim"] == 128
    elif expected == "global":
        from warpack._projected_cholesky import global_cholesky

        assert calls[0][1][0] is global_cholesky
        assert calls[0][2]["inputs"] == [*arguments, work]
        assert calls[0][2]["dim"] == 1
        assert calls[0][2]["block_dim"] == (256 if size <= 256 else 512)
    else:
        assert calls[0][1] == (K.cholesky, 1, arguments)
    # Other operations must continue through the original launch path.
    calls.clear()
    BlockLOBPCG.launch(owner, K.triangular, 3, arguments)
    assert calls[0][0] == "serial" and calls[0][1] == (K.triangular, 3, arguments)


def test_explicit_serial_benchmark_does_not_follow_new_default(monkeypatch):
    owner = SimpleNamespace(
        device=SimpleNamespace(is_cuda=True, max_shared_memory_per_block=100000)
    )
    arguments = [SimpleNamespace(shape=(104, 104)), object(), object()]
    calls = []
    monkeypatch.setattr(wp, "launch", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(wp, "launch_tiled", lambda *a, **kw: pytest.fail("serial control changed"))
    SerialCholeskyBlock.launch(owner, K.cholesky, 1, arguments)
    assert calls == [(K.cholesky, 1, arguments)]
