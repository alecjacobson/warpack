"""Truncated preconditioning still needs complete original finite eigenspaces."""

import numpy as np
import pytest
import warp as wp
import warp.optim.linear as linear

from benchmarks.applications.conformal import boundary_reference, make_conformal_case, metrics
from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.conformal_block import ConformalBlockEigensolver

supported = hasattr(linear, "FSAI") and hasattr(linear.LinearOperator, "batch_offsets")
requires_upstream = pytest.mark.skipif(
    not supported, reason="requires pinned upstream FSAI/batched CG"
)


def test_explicit_unsupported_runtime(monkeypatch):
    monkeypatch.delattr(linear, "FSAI", raising=False)
    with pytest.raises(RuntimeError, match="pinned upstream"):
        ConformalBlockEigensolver(object())


@requires_upstream
@pytest.mark.parametrize("height", [0.0, 0.25])
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("modes,width", [(2, 4), (6, 8)])
def test_changed_starts_and_original_modes(height, dtype, modes, width, tmp_path):
    case = make_conformal_case(12, modes, height, shift=-0.001)
    reference = boundary_reference(case)
    solver = ConformalBlockEigensolver(case, width, 20, working_dtype=dtype)
    graph = solver.capture(200)
    try:
        rng = np.random.default_rng(473)
        for _ in range(2):
            solver.core.initial_vectors.assign(rng.normal(size=(width, case.n)))
            solver.inverse.reset_stats()
            wp.capture_launch(graph)
            check = metrics(
                case, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, reference
            )
            assert check["accepted"], check
            assert solver.core.converged.numpy()[0] == modes
            assert solver.converged.numpy()[0] == modes
            assert solver.core.status.numpy()[0] == 0
            assert solver.inverse.stats.numpy()[3] == width * solver.core.iterations.numpy()[0]
            gamma = solver.core.eigenvalues.numpy()
            assert np.all(gamma < 0)
            np.testing.assert_allclose(
                solver.eigenvalues.numpy(), case.shift + case.shift / gamma, atol=1e-15
            )
            # Repeated recovery must not transform the core Ritz values in place.
            solver.finalize()
            np.testing.assert_array_equal(solver.core.eigenvalues.numpy(), gamma)
        dot = tmp_path / "conformal-block.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
        assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
    finally:
        graph = None
        solver.close()


@requires_upstream
def test_short_block_budget_is_rejected():
    case = make_conformal_case(16, 6, 0.25, shift=-0.001)
    solver = ConformalBlockEigensolver(case, 8, 10)
    graph = solver.capture(1)
    try:
        wp.capture_launch(graph)
        check = metrics(
            case,
            solver.eigenvalues.numpy(),
            solver.eigenvectors.numpy().T,
            boundary_reference(case),
        )
        assert not check["accepted"]
        assert solver.core.converged.numpy()[0] < case.k
    finally:
        graph = None
        solver.close()


@requires_upstream
def test_nonfinite_start_then_same_graph_recovers():
    case = make_conformal_case(12, 6, 0.25, shift=-0.001)
    solver = ConformalBlockEigensolver(case, 8, 20)
    good = solver.core.initial_vectors.numpy()
    graph = solver.capture(200)
    try:
        solver.core.initial_vectors.fill_(float("nan"))
        wp.capture_launch(graph)
        assert solver.core.status.numpy()[0] != 0
        solver.core.initial_vectors.assign(good)
        wp.capture_launch(graph)
        check = metrics(
            case,
            solver.eigenvalues.numpy(),
            solver.eigenvectors.numpy().T,
            boundary_reference(case),
        )
        assert check["accepted"], check
        assert solver.converged.numpy()[0] == case.k
        assert solver.core.status.numpy()[0] == 0
    finally:
        graph = None
        solver.close()


@requires_upstream
def test_native_cupy_truncated_cg_reference():
    cp = pytest.importorskip("cupy")
    from benchmarks.applications.conformal_backends import cupy_cg_reciprocal

    case = make_conformal_case(12, 6, 0.25, shift=-0.001)
    reference = boundary_reference(case)
    solve, close = cupy_cg_reciprocal(case, inner_steps=25, tolerance=1e-10)
    try:
        for _ in range(2):
            values, vectors, (history, fresh, stats) = solve()
            check = metrics(case, values.get(), vectors.get(), reference)
            assert check["accepted"], check
            assert float(cp.max(history[-1])) <= 1e-10
            assert float(cp.max(fresh)) <= 1e-10
            iterations, maximum, unfinished, rhs_calls = stats.numpy()
            assert 0 < maximum <= 25
            assert 0 < iterations <= 25 * rhs_calls
            assert 0 <= unfinished <= rhs_calls
            # Shrinking native active blocks must not be padded to k columns.
            assert 0 < rhs_calls < case.k * (len(history) - 1)
    finally:
        close()
