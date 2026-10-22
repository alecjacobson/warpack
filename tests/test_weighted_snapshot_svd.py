"""Physical weighted-SVD modes, weak directions and graph failure recovery."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from benchmarks.applications.fem_hierarchy import audit, beam_problem
from benchmarks.applications.force_dual import SnapshotROM, localized_forces, reference
from benchmarks.applications.weighted_snapshot_reference import weighted_reference
from benchmarks.applications.weighted_snapshot_svd import LumpedSnapshotSVD
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("weak", [1e-4, 1e-6])
def test_weak_physical_modes_and_changed_mass_replay(weak, tmp_path):
    rng = np.random.default_rng(13504)
    n, r, k = 96, 12, 8
    q = np.linalg.qr(rng.normal(size=(n, k)))[0]
    v = np.linalg.qr(rng.normal(size=(r, k)))[0]
    spectrum = np.r_[np.linspace(2, 0.4, k - 1), weak]
    mass = np.linspace(0.5, 3, n)
    g = ((q / np.sqrt(mass)[:, None]) * spectrum) @ v.T
    rows = wp.array(g.T.copy(), dtype=wp.float64)
    weights = wp.array(mass, dtype=wp.float64)
    driver = LumpedSnapshotSVD(rows, weights, k, tol=1e-14, ncv=64)
    driver.capture(100)
    for current in (mass, mass[::-1].copy(), mass):
        weights.assign(current)
        driver.solve()
        ref = weighted_reference(g, sp.diags(current), k)
        assert driver.status.numpy()[0] == 0
        assert driver.positive.numpy()[0] == k
        u, values = driver.modes.numpy().T, driver.values.numpy()
        np.testing.assert_allclose(values, ref["values"], rtol=1e-7, atol=0)
        np.testing.assert_allclose(u.T @ (current[:, None] * u), np.eye(k), atol=1e-10)
        remainder = u - ref["basis"] @ (ref["basis"].T @ (current[:, None] * u))
        assert np.sqrt(np.sum(remainder**2 * current[:, None])) < 1e-8
        captured = np.sum((u.T @ (current[:, None] * g)) ** 2)
        np.testing.assert_allclose(captured, sum(ref["values"]), rtol=1e-12)
    if weak == 1e-6:
        # The public augmented SVD resolves a positive mode the Gram cutoff excludes.
        assert spectrum[-1] ** 2 < 1e-12 * spectrum[0] ** 2
    for i, graph in enumerate(driver.graphs):
        audit(graph, tmp_path / f"weighted-{i}.dot")


def test_rank_invalid_mass_and_nonfinite_response_recover_on_same_graphs(tmp_path):
    rng = np.random.default_rng(13505)
    mass = np.linspace(1, 2, 40)
    g = rng.normal(size=(40, 10))
    rows, weights = wp.array(g.T.copy(), dtype=wp.float64), wp.array(mass, dtype=wp.float64)
    driver = LumpedSnapshotSVD(rows, weights, 4, ncv=40)
    driver.capture(100)
    for kind in ("valid", "rank2", "zero", "negative", "nan", "valid"):
        current, m = g.copy(), mass.copy()
        if kind == "rank2":
            current[:, 2:] = 0
        elif kind == "zero":
            current[:] = 0
        elif kind == "negative":
            m[-1] = -1
        elif kind == "nan":
            current[0, 0] = np.nan
        rows.assign(current.T.copy())
        weights.assign(m)
        driver.solve()
        status, positive = driver.status.numpy()[0], driver.positive.numpy()[0]
        if kind == "valid":
            assert status == 0 and positive == 4
            ref = weighted_reference(current, sp.diags(m), 4)
            np.testing.assert_allclose(driver.values.numpy(), ref["values"], rtol=1e-10)
        else:
            assert status != 0 and positive < 4
        assert np.isfinite(driver.modes.numpy()).all()
        assert np.isfinite(driver.values.numpy()).all()
    for i, graph in enumerate(driver.graphs):
        audit(graph, tmp_path / f"weighted-recovery-{i}.dot")


@pytest.mark.parametrize("backend", ["cudss", "cg"])
def test_complete_force_response_and_svd_workflow(backend, tmp_path):
    from warp.optim import linear

    if backend == "cg" and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(2, 4, consistent=False)
    f = localized_forces(2, 12)
    forces = wp.array(f.T.copy(), dtype=wp.float64)
    driver = SnapshotROM(
        matrix(p.a),
        matrix(p.b),
        forces,
        4,
        backend,
        diagonal_mass=wp.array(p.b.diagonal(), dtype=wp.float64),
    )
    try:
        driver.capture()
        assert all(g.graph_exec is not None for g in driver.mode_graphs)
        # Change the input before the first caller replay: setup's initialized
        # outputs must not accidentally hide stale-response or first-upload bugs.
        for scale in (np.linspace(0.4, 1.7, 12), np.ones(12), np.linspace(1.8, 0.3, 12)):
            current = f * scale
            forces.assign(current.T.copy())
            wp.capture_launch(driver.response_graph)
            driver.solve_modes()
            quality = driver.quality(
                p.a, p.b, current, reference(p.a, p.b, current, 4, method="svd")
            )
            assert quality["accepted"], quality
        for name, graph in driver.audit_graphs().items():
            if name != "response" or backend == "cg":
                audit(graph, tmp_path / (name + ".dot"))
    finally:
        driver.close()
