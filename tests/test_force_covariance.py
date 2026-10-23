"""Full-rank implicit force priors and mixed inverse backends preserve physical modes."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import beam_problem
from benchmarks.applications.force_covariance import ForceCovarianceSolver, dense_reference
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("prior_kind", ["mass", "diagonal"])
@pytest.mark.parametrize("backend,metric", [("cudss", "cudss"), ("cg", "cg"), ("cudss", "cg")])
def test_full_rank_prior_modes_and_changed_prior_replay(
    consistent, prior_kind, backend, metric, tmp_path
):
    if "cg" in (backend, metric) and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(2, 4, consistent)
    prior = (
        p.b.copy()
        if prior_kind == "mass"
        else sp.diags(p.b.diagonal() * np.linspace(0.2, 2, p.n), format="csr")
    )
    hp, mp, spw = matrix(p.a), matrix(p.b), matrix(prior)
    driver = ForceCovarianceSolver(hp, mp, spw, 4, backend, metric_backend=metric, ncv=24)
    try:
        if backend == "cudss" and metric == "cg":
            assert driver.solver.iter_op.requires_static_outer_capture
        driver.capture(4)
        # Probe the actual composed operator, independently of its eigenvectors.
        # Strict CG stopping alone does not establish numerical self-adjointness.
        probe = np.random.default_rng(13515).normal(size=(3, p.n))
        probe[1] *= 1e-5
        probe[2] *= 1e5
        x = wp.array(probe, dtype=wp.float64)
        ax = wp.empty_like(x)
        driver.operator.apply(x, ax)
        observed = ax.numpy()
        dense = dense_reference(p.a, p.b, prior, 4)["a"]
        expected = (dense @ probe.T).T
        assert (
            np.max(np.linalg.norm(observed - expected, axis=1) / np.linalg.norm(expected, axis=1))
            < 1e-10
        )
        products = probe @ observed.T
        scale = np.linalg.norm(probe, axis=1)[:, None] * np.linalg.norm(observed, axis=1)[None, :]
        assert np.max(abs(products - products.T) / np.maximum(scale, scale.T)) < 1e-10
        d = sp.diags(np.linspace(0.7, 1.4, p.n))
        for current in (d @ prior @ d, prior):
            changed = matrix(current)
            wp.copy(spw.values, changed.values)
            driver.solve()
            values, u = driver.solver.eigenvalues.numpy(), driver.solver.eigenvectors.numpy().T
            ref = dense_reference(p.a, p.b, current, 4)
            assert driver.solver.converged.numpy()[0] == 4
            assert driver.solver.status.numpy()[0] == 0
            np.testing.assert_allclose(values, ref["values"], rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(u.T @ (p.b @ u), np.eye(4), atol=1e-10)
            residual = ref["a"] @ u - (p.b @ u) * values
            assert (
                np.max(
                    np.linalg.norm(residual, axis=0)
                    / np.maximum(
                        1,
                        np.linalg.norm(ref["a"] @ u, axis=0)
                        + abs(values) * np.linalg.norm(p.b @ u, axis=0),
                    )
                )
                < 1e-10
            )
            remainder = u - ref["basis"] @ (ref["basis"].T @ (p.b @ u))
            assert np.sqrt(np.sum(remainder * (p.b @ remainder))) < 1e-8
            np.testing.assert_allclose(np.trace(u.T @ ref["a"] @ u), sum(ref["values"]), rtol=1e-10)
            for inverse in driver.inverses:
                if hasattr(inverse, "stats"):
                    assert inverse.stats.numpy()[2] == 0
                    assert inverse.residual_maxima.numpy()[1] < 1e-9
        if prior_kind == "mass":
            modal = la.eigh(p.a.toarray(), p.b.toarray(), eigvals_only=True)[:4]
            np.testing.assert_allclose(values, 1 / modal**2, rtol=1e-9)
        path = tmp_path / "implicit.dot"
        wp.capture_debug_dot_print(driver.graph, str(path))
        audit = graph_summary(path)
        assert not audit["node_types_including_conditional_body"].get("HOST", 0)
        assert not any("DtoH" in x for x in audit["memcpy_kinds"])
        if backend == metric == "cg":
            assert not any("HtoD" in x for x in audit["memcpy_kinds"])
    finally:
        driver.close()


@pytest.mark.parametrize("backend", ["cudss", "cg"])
def test_singular_force_prior_rank_exhaustion_and_recovery(backend):
    if backend == "cg" and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(1, 4, False)
    prior = matrix(p.b)
    original = prior.values.numpy().copy()
    driver = ForceCovarianceSolver(matrix(p.a), matrix(p.b), prior, 4, backend, ncv=24)
    try:
        driver.capture(4)
        for rank in (2, 0, p.n):
            current = original.copy()
            current[rank:] = 0
            prior.values.assign(current)
            driver.solve()
            values = driver.solver.eigenvalues.numpy()
            u = driver.solver.eigenvectors.numpy().T
            assert np.isfinite(values).all() and np.isfinite(u).all()
            positive = np.count_nonzero(values > max(0, values[0]) * 1e-12)
            if rank == 0:
                assert np.max(abs(values)) < 1e-12
                assert positive == 0
            elif rank == 2:
                assert positive == 2  # Four positive ROM modes do not exist.
                ref = dense_reference(p.a, p.b, sp.diags(current), 2)
                np.testing.assert_allclose(values[:2], ref["values"], rtol=1e-9)
            else:
                assert positive == 4
                assert driver.solver.status.numpy()[0] == 0
                assert driver.solver.converged.numpy()[0] == 4
                ref = dense_reference(p.a, p.b, p.b, 4)
                np.testing.assert_allclose(values, ref["values"], rtol=1e-9)
    finally:
        driver.close()


def test_explicit_fixed_cg_budget_preserves_original_problem():
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(1, 4, False)
    driver = ForceCovarianceSolver(matrix(p.a), matrix(p.b), matrix(p.b), 4, "cg", ncv=p.n)
    try:
        driver.capture(1, adaptive=False)
        assert driver.adaptive is False
        driver.solve()
        assert driver.solver.iterations.numpy()[0] == 1
        assert driver.solver.converged.numpy()[0] == 4
        ref = dense_reference(p.a, p.b, p.b, 4)
        np.testing.assert_allclose(driver.solver.eigenvalues.numpy(), ref["values"], rtol=1e-9)
    finally:
        driver.close()
