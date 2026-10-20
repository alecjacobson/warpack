"""LA sign transform preserves the original pencil, including changed replay."""

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
import warp.optim.linear as linear

from benchmarks.experimental.largest_block import LargestBlock
from benchmarks.upstream.adapters import matrix


@pytest.mark.parametrize("metric", ["identity", "diagonal", "coupled"])
def test_largest_block_is_la_not_lm_and_refreshes_coefficients(metric, tmp_path):
    n, k, width = 48, 4, 8
    rng = np.random.default_rng(841)
    q = la.qr(rng.normal(size=(n, n)))[0]
    spectrum = np.linspace(-10, 10, n)
    spectrum[:2] = [-200, -150]  # LM must differ from LA.
    spectrum[-4:] = [11, 11, 12, 12]
    lower = np.eye(n)
    if metric != "identity":
        lower = np.diag(np.linspace(0.7, 1.3, n))
    if metric == "coupled":
        lower += np.diag(np.full(n - 1, 0.15), -1)
    mass = lower @ lower.T
    modes = la.solve_triangular(lower.T, q, lower=False)
    dense = mass @ (modes * spectrum) @ modes.T @ mass
    a = matrix(dense)
    b = matrix(mass)
    starts = wp.array(rng.normal(size=(width, n)), dtype=wp.float64)
    solver = LargestBlock(
        a,
        k,
        mass=b,
        width=width,
        initial_vectors=starts,
        preconditioner=linear.preconditioner(b, "diag"),
        tol=1e-10,
    )
    graph = solver.capture(200)
    for factor, shift in [(1.0, 0.0), (1.5, -0.5), (1.0, 0.0)]:
        host = factor * dense + shift * mass
        changed = matrix(host)
        np.testing.assert_array_equal(a.offsets.numpy(), changed.offsets.numpy())
        np.testing.assert_array_equal(a.columns.numpy(), changed.columns.numpy())
        wp.copy(a.values, changed.values)
        initial = rng.normal(size=starts.shape)
        starts.assign(initial)
        wp.capture_launch(graph)
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        expected, basis = la.eigh(host, mass)
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        np.testing.assert_allclose(w, expected[-k:][::-1], atol=1e-9, rtol=0)
        np.testing.assert_allclose(v.T @ mass @ v, np.eye(k), atol=1e-10)
        assert la.norm(host @ v - (mass @ v) * w, axis=0).max() < 1e-8
        wanted = basis[:, -k:]
        assert la.norm(v - wanted @ (wanted.T @ mass @ v)) < 1e-8
        np.testing.assert_array_equal(starts.numpy(), initial)
        np.testing.assert_array_equal(a.values.numpy(), changed.values.numpy())
    dot = tmp_path / "largest-block.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    assert not any(token in dot.read_text() for token in ("HOST", "HtoD", "DtoH"))


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("side", [3, 5])
def test_upper_q1_preconditioner_matrix_is_spd_and_preserves_inputs(consistent, side):
    import scipy.sparse as sp

    from benchmarks.applications.negative_foundation import make_negative_foundation
    from benchmarks.experimental.largest_block import upper_shifted_matrix

    case = make_negative_foundation(side, 4, "LM", consistent)
    p = case.problem
    upper = (60.0 if consistent else 20.0) * (side + 1) ** 2 - case.foundation
    # Independent dense generalized spectrum verifies the stencil bound.
    assert la.eigvalsh(p.a.toarray(), p.b.toarray()).max() < upper
    a, b = matrix(p.a), matrix(p.b)
    old_a, old_b = a.values.numpy(), b.values.numpy()
    shifted = upper_shifted_matrix(a, b, upper)
    offsets = shifted.offsets.numpy()
    count = offsets[-1]
    actual = sp.csr_matrix(
        (shifted.values.numpy()[:count], shifted.columns.numpy()[:count], offsets), shape=p.a.shape
    ).toarray()
    np.testing.assert_allclose(actual, (upper * p.b - p.a).toarray(), rtol=1e-14, atol=1e-14)
    assert la.eigvalsh(actual).min() > 0
    np.testing.assert_array_equal(a.values.numpy(), old_a)
    np.testing.assert_array_equal(b.values.numpy(), old_b)
