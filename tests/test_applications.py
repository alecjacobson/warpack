"""Application contracts: independent physics/topology and original-pencil checks."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp

from benchmarks.applications.problems import bar, make_problem, torus_hodge
from benchmarks.applications.runner import accuracy, warp_output, warp_setup


def test_accuracy_gate_rejects_wrong_requested_eigenpairs():
    p = make_problem("A", 31)
    w, v = la.eigh(p.a.toarray(), p.b.toarray())
    wrong = accuracy(p, w[-p.k :], v[:, -p.k :])
    assert wrong["residual"] < 1e-12
    assert wrong["orthogonality"] < 1e-12
    assert not wrong["accepted"]
    assert accuracy(p, w[: p.k], v[:, : p.k])["accepted"]


def test_regression_gate_handles_contention_and_worker_failure():
    from benchmarks.applications.compare import compare_baseline

    good = dict(
        api_class="A",
        n=10000,
        requested_size=10000,
        method="warp",
        accepted=True,
        timing_uncontended=True,
        solve_seconds=1.0,
        device_only_graph=True,
        matrix_sha256="fixture",
        k=4,
        ncv=32,
        outer_cycles=5,
    )
    baseline = dict(cases=[good])
    slow = compare_baseline([{**good, "solve_seconds": 2.0}], baseline, 1.25)[0]
    assert slow["performance_regression"]
    busy = compare_baseline([{**good, "timing_uncontended": False}], baseline, 1.25)[0]
    assert not busy["comparable"]
    changed_tol = compare_baseline(
        [{**good, "reference_tolerance": 1e-10}],
        {"cases": [{**good, "reference_tolerance": 1e-8}]},
        1.25,
    )[0]
    assert not changed_tol["comparable"]
    changed_inner = compare_baseline(
        [{**good, "cg_tolerance": 1e-15}], {"cases": [{**good, "cg_tolerance": 1e-13}]}, 1.25
    )[0]
    assert not changed_inner["comparable"]
    changed_shift = compare_baseline(
        [{**good, "target": [-1.0, 0.0]}],
        {"cases": [{**good, "target": [-0.01, 0.0]}]},
        1.25,
    )[0]
    assert not changed_shift["comparable"]
    unchanged_projection = compare_baseline([{**good, "projected_rigid_modes": 0}], baseline, 1.25)[
        0
    ]
    assert unchanged_projection["comparable"]
    changed_projection = compare_baseline([{**good, "projected_rigid_modes": 6}], baseline, 1.25)[0]
    assert not changed_projection["comparable"]
    failed = dict(api_class="A", requested_size=10000, method="warp", accepted=False)
    assert compare_baseline([failed], baseline, 1.25)[0]["accuracy_regression"]


def test_spectra_generalized_shift_regression():
    """Issue #7: a small physical residual must accompany the requested spectrum."""
    from pathlib import Path

    from benchmarks.upstream.adapters import matrix
    from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

    data = np.load(Path(__file__).parent / "fixtures/spectra_generalized_shift_5.npz")
    a, b = data["a"], data["b"]
    sigma, k = 1.2345, 10
    expected, basis = la.eigh(a, b)
    selected = np.argsort(abs(expected - sigma))[:k]
    inverse = CuDSSInverse(matrix(a - sigma * b), 1, mtype="symmetric")
    solver = GeneralizedKrylovSchur(
        SparseOperator(matrix(a)),
        SparseOperator(matrix(b)),
        inverse,
        k,
        ncv=20,
        target=sigma,
        tol=1e-12,
    )
    graph = solver.capture(20)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            v, w = solver.eigenvectors.numpy().T, solver.eigenvalues.numpy()
            np.testing.assert_allclose(np.sort(w), np.sort(expected[selected]), atol=1e-12)
            np.testing.assert_allclose(v.T @ b @ v, np.eye(k), atol=1e-12)
            assert np.max(abs(a @ v - (b @ v) * w)) < 1e-12
            # Compare the invariant subspace, without requiring matching vector signs.
            overlap = basis[:, selected].T @ b @ v
            np.testing.assert_allclose(la.svdvals(overlap), np.ones(k), atol=1e-12)
            assert solver.converged.numpy()[0] == k
    finally:
        solver.graph = None
        del graph, solver
        inverse.close()


@pytest.mark.parametrize("api_class", list("ABCDEFGHI"))
def test_application_graph_replay(api_class):
    p = make_problem(api_class, 80)
    solver, inverse = warp_setup(p)
    steps = 100 if api_class in "DG" else 40
    graph = solver.capture(steps)
    try:
        for _ in range(2):
            wp.capture_launch(graph)
            result = accuracy(p, *warp_output(p, solver))
            assert result["accepted"], (api_class, result)
    finally:
        del graph, solver
        if inverse is not None:
            inverse.close()


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("free", [False, True])
def test_fem_fixture_analytic_spectrum(consistent, free):
    a, b, w = bar(31, consistent=consistent, free=free)
    np.testing.assert_allclose(la.eigvalsh(a.toarray(), b.toarray()), w, atol=1e-10)


def test_torus_fixture_independent_boundary_ranks():
    a, b1, b2 = torus_hodge(4)
    assert (b1 @ b2).nnz == 0
    beta = a.shape[0] - np.linalg.matrix_rank(b1.toarray()) - np.linalg.matrix_rank(b2.toarray())
    assert beta == 2
    vals = la.eigvalsh(a.toarray())
    assert np.count_nonzero(abs(vals) < 1e-10) == beta
    assert vals[beta] > 0.1


def test_generalized_regular_mode_and_updated_start():
    from benchmarks.upstream.adapters import matrix
    from warpack import CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

    a, b, w = bar(60)
    inverse = CuDSSInverse(matrix(b), 1)
    start = wp.ones(60, dtype=wp.float64)
    solver = GeneralizedKrylovSchur(
        SparseOperator(matrix(a)),
        SparseOperator(matrix(b)),
        inverse,
        4,
        ncv=32,
        mode="regular",
        which="LA",
        initial_vector=start,
        tol=1e-10,
    )
    graph = solver.capture(30)
    try:
        for seed in [11, 12]:
            wp.copy(start, wp.array(np.random.default_rng(seed).normal(size=60), dtype=wp.float64))
            wp.capture_launch(graph)
            v = solver.eigenvectors.numpy().T
            values = solver.eigenvalues.numpy()
            np.testing.assert_allclose(np.sort(values), w[-4:], rtol=1e-10)
            np.testing.assert_allclose(v.T @ b @ v, np.eye(4), atol=1e-10)
            assert solver.residuals.numpy().max() < 1e-10
    finally:
        del graph, solver
        inverse.close()
