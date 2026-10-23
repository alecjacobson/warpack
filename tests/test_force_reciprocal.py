"""Both physical covariance and reciprocal equations gate the alternative recipe."""

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_hierarchy import beam_problem
from benchmarks.applications.force_covariance import dense_reference
from benchmarks.applications.force_reciprocal import ForceReciprocalSolver
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize(
    "backend,refinement,polish",
    [("cudss", 0, 0), ("cg", 0, 0), ("cg", 1, 0), ("cg", 0, 1), ("cudss", 0, 1)],
)
def test_reciprocal_force_modes_changed_prior_and_both_physical_equations(
    consistent, backend, refinement, polish, tmp_path
):
    if backend == "cg" and not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    p = beam_problem(2, 4, consistent)
    d = p.b.diagonal() * np.linspace(0.2, 2, p.n)
    prior, prior_inverse = matrix(sp.diags(d)), matrix(sp.diags(1 / d))
    driver = ForceReciprocalSolver(
        matrix(p.a),
        matrix(p.b),
        prior,
        prior_inverse,
        4,
        backend,
        ncv=32,
        refinement_steps=refinement,
        polish_steps=polish,
    )
    try:
        stages = driver.capture(4)
        assert stages == tuple(
            g for g in (driver.graph, driver.polish_graph, driver.recovery_graph) if g is not None
        )
        probe = np.random.default_rng(152).normal(size=(3, p.n))
        probe[1] *= 1e-5
        probe[2] *= 1e5
        x = wp.array(probe, dtype=wp.float64)
        y = wp.empty_like(x)
        for current in (d * np.linspace(0.7, 1.4, p.n), d):
            wp.copy(prior.values, matrix(sp.diags(current)).values)
            wp.copy(prior_inverse.values, matrix(sp.diags(1 / current)).values)
            # First caller replay already has changed values in both operators.
            driver.solve()
            values, u = driver.eigenvalues.numpy(), driver.eigenvectors.numpy().T
            mu = driver.solver.eigenvalues.numpy()
            ref = dense_reference(p.a, p.b, sp.diags(current), 4)
            q = p.a @ sp.diags(1 / current) @ p.a
            dense_mu = la.eigh(q.toarray(), p.b.toarray(), eigvals_only=True)[:4]
            assert driver.solver.status.numpy()[0] == 0
            assert driver.solver.converged.numpy()[0] == 4
            np.testing.assert_allclose(mu, dense_mu, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(values, ref["values"], rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(u.T @ (p.b @ u), np.eye(4), atol=1e-10)
            remainder = u - ref["basis"] @ (ref["basis"].T @ (p.b @ u))
            assert np.sqrt(np.sum(remainder * (p.b @ remainder))) < 1e-8
            for operator, lam in ((q, mu), (ref["a"], values)):
                au, mu_basis = operator @ u, p.b @ u
                relative = np.linalg.norm(au - mu_basis * lam, axis=0) / np.maximum(
                    1, np.linalg.norm(au, axis=0) + abs(lam) * np.linalg.norm(mu_basis, axis=0)
                )
                assert max(relative) < 1e-9
            np.testing.assert_allclose(np.trace(u.T @ ref["a"] @ u), sum(ref["values"]), rtol=1e-10)
            # Check each sandwich itself, independently of spectral convergence.
            hi = la.cho_solve(la.cho_factor(p.a.toarray()), np.eye(p.n))
            covariance = (hi * current) @ hi
            for operator, expected_matrix in (
                (driver.operator, q),
                (driver.covariance, covariance),
            ):
                operator.apply(x, y)
                observed, expected = y.numpy(), (expected_matrix @ probe.T).T
                assert (
                    max(
                        np.linalg.norm(observed - expected, axis=1)
                        / np.linalg.norm(expected, axis=1)
                    )
                    < 1e-10
                )
                products = probe @ observed.T
                scale = (
                    np.linalg.norm(probe, axis=1)[:, None]
                    * np.linalg.norm(observed, axis=1)[None, :]
                )
                assert np.max(abs(products - products.T) / np.maximum(scale, scale.T)) < 1e-10
            if backend == "cg":
                assert driver.inverse.stats.numpy()[2] == 0
                assert driver.inverse.residual_maxima.numpy()[1] < 1e-9
        for name, stage in (
            ("solve", driver.graph),
            ("polish", driver.polish_graph),
            ("recover", driver.recovery_graph),
        ):
            if stage is None:
                continue
            dot = tmp_path / f"{name}.dot"
            wp.capture_debug_dot_print(stage, str(dot))
            graph = graph_summary(dot)
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert not any("DtoH" in kind for kind in graph["memcpy_kinds"])
            if backend == "cg":
                assert not any("HtoD" in kind for kind in graph["memcpy_kinds"])
    finally:
        driver.close()
