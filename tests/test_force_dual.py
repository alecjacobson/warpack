"""Independent covariance, modal equivalence, energy and rank-exhaustion checks."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import beam_problem
from benchmarks.applications.force_dual import SnapshotROM, localized_forces, reference
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("backend", ["cudss", "cg"])
@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("dependent", [False, True])
def test_local_forces_and_changed_covariance_replay(backend, consistent, dependent, tmp_path):
    if backend == "cg" and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(2, 4, consistent)
    forces = localized_forces(2, 12)
    if dependent:
        forces[:, -4:] = forces[:, :4] @ np.diag([1.0, 2.0, -1.0, 0.5])
    device = wp.array(forces.T.copy(), dtype=wp.float64)
    driver = SnapshotROM(matrix(p.a), matrix(p.b), device, 4, backend)
    try:
        driver.capture()
        for scale in (np.ones(12), np.linspace(0.3, 2.0, 12)):
            current = forces * scale
            ref = reference(p.a, p.b, current, 4)
            device.assign(current.T.copy())
            wp.capture_launch(driver.response_graph)
            wp.capture_launch(driver.mode_graph)
            q = driver.quality(p.a, p.b, current, ref)
            assert q["accepted"], q
            # A distinct full covariance reference checks the Gram reduction.
            covariance = ref["snapshots"] @ ref["snapshots"].T
            vals = la.eigvalsh(p.b @ covariance @ p.b, p.b.toarray())[-4:][::-1]
            np.testing.assert_allclose(driver.solver.eigenvalues.numpy(), vals, rtol=1e-10)
        for name, graph in (("response", driver.response_graph), ("modes", driver.mode_graph)):
            path = tmp_path / (name + ".dot")
            wp.capture_debug_dot_print(graph, str(path))
            text = path.read_text()
            assert "HOST" not in graph_summary(path)["node_types_including_conditional_body"]
            if backend == "cg" or name == "modes":
                assert "HtoD" not in text and "DtoH" not in text
    finally:
        driver.close()


@pytest.mark.parametrize("consistent", [False, True])
def test_mass_force_covariance_recovers_modal_subspace(consistent):
    p = beam_problem(1, 6, consistent)
    forces = la.cholesky(p.b.toarray(), lower=True)
    ref = reference(p.a, p.b, forces, 6)
    modal_values, modal_vectors = la.eigh(p.a.toarray(), p.b.toarray())
    np.testing.assert_allclose(ref["values"], 1 / modal_values[:6] ** 2, rtol=1e-10)
    driver = SnapshotROM(matrix(p.a), matrix(p.b), wp.array(forces.T.copy(), dtype=wp.float64), 6)
    try:
        driver.capture()
        wp.capture_launch(driver.response_graph)
        wp.capture_launch(driver.mode_graph)
        q = driver.quality(p.a, p.b, forces, ref)
        assert q["accepted"], q
        u = driver.vectors.numpy().T
        remainder = u - modal_vectors[:, :6] @ (modal_vectors[:, :6].T @ (p.b @ u))
        assert np.sqrt(np.sum(remainder * (p.b @ remainder))) < 1e-8
    finally:
        driver.close()


def test_rank_exhaustion_rejects_and_recovers_on_same_graph():
    p = beam_problem(1, 4)
    forces = localized_forces(1, 8)
    device = wp.array(forces.T.copy(), dtype=wp.float64)
    driver = SnapshotROM(matrix(p.a), matrix(p.b), device, 4)
    try:
        driver.capture()
        for rank in (8, 2, 0, 8):
            current = forces.copy()
            current[:, rank:] = 0
            device.assign(current.T.copy())
            wp.capture_launch(driver.response_graph)
            wp.capture_launch(driver.mode_graph)
            ref = reference(p.a, p.b, current, 4)
            q = driver.quality(p.a, p.b, current, ref)
            assert q["accepted"] == (rank >= 4), q
            assert np.isfinite(driver.vectors.numpy()).all()
            assert q["positive"] == min(4, rank)
    finally:
        driver.close()
