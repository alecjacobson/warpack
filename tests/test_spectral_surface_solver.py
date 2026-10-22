"""Dominant generalized modes with a nonmass metric and complete repeated clusters."""

import numpy as np
import pytest
import warp as wp
from warp.optim import linear

from benchmarks.applications.runner import accuracy
from benchmarks.applications.spectral_surface import surface_pencil
from benchmarks.applications.spectral_surface_compare import SurfaceSolver, problem
from benchmarks.audit_profile import graph_summary


@pytest.mark.parametrize("backend", ["cg", "cudss"])
@pytest.mark.parametrize("copies", [1, 2])
def test_dominant_cluster_and_changed_anisotropy_replay(backend, copies, tmp_path):
    if backend == "cg" and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = problem(surface_pencil(4), copies, copies)
    driver = SurfaceSolver(p, backend, ncv=40)
    try:
        driver.capture(4)
        for anisotropy in (10, 30, 5):
            current = problem(surface_pencil(4, anisotropy=anisotropy), copies, copies)
            np.testing.assert_array_equal(p.a.indptr, current.a.indptr)
            np.testing.assert_array_equal(p.a.indices, current.a.indices)
            np.testing.assert_array_equal(p.b.toarray(), current.b.toarray())
            driver.a.values[: current.a.nnz].assign(current.a.data)
            driver.solve()
            values, vectors = driver.output()
            q = accuracy(current, values, vectors)
            assert q["accepted"], q
            assert driver.solver.status.numpy()[0] == 0
            assert driver.solver.converged.numpy()[0] == copies
            assert q["residual"] < 1e-11
        path = tmp_path / "surface.dot"
        wp.capture_debug_dot_print(driver.graph, str(path))
        graph = graph_summary(path)
        assert "HOST" not in graph["node_types_including_conditional_body"]
        assert not any(k.startswith("DtoH") for k in graph["memcpy_kinds"])
        if backend == "cg":
            assert not any(k.startswith("HtoD") for k in graph["memcpy_kinds"])
    finally:
        driver.close()


@pytest.mark.parametrize("alpha,gamma", [(10.0, 0.0), (1000.0, 0.01)])
def test_regularization_metric_changes(alpha, gamma):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = problem(surface_pencil(4, alpha=alpha, gamma=gamma), 3)
    driver = SurfaceSolver(p, "cg", ncv=40)
    try:
        driver.capture(6)
        driver.solve()
        q = accuracy(p, *driver.output())
        assert q["accepted"], q
        assert driver.solver.converged.numpy()[0] == 3
    finally:
        driver.close()


@pytest.mark.parametrize("backend", ["cg", "cudss"])
def test_sampling_changes_rebuild_regularization_metric(backend):
    if backend == "cg" and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    previous = None
    for samples in (64, 160):
        p = problem(surface_pencil(4, sample_count=samples))
        if previous is not None:
            assert np.linalg.norm((p.b - previous).data) > 0.1
        previous = p.b
        driver = SurfaceSolver(p, backend)
        try:
            driver.capture(4)
            driver.solve()
            q = accuracy(p, *driver.output())
            assert q["accepted"], q
            assert driver.solver.converged.numpy()[0] == 1
        finally:
            driver.close()
