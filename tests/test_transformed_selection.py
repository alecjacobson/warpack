"""Physical target mapping and achievable profiles for inverse-spectrum SM."""

import re
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from benchmarks.upstream.run_spectra import select_symmetric_reference
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator


@pytest.mark.parametrize(
    "which,expected",
    [
        ("LM", [0.9, 1.1]),
        ("SM", [-10.0, 6.0]),
        ("LA", [1.1, 3.0]),
        ("SA", [-1.0, 0.9]),
        ("BE", [0.9, 1.1]),
    ],
)
def test_inverse_selection_uses_transformed_order(which, expected):
    values = np.array([-10.0, -1.0, 0.9, 1.1, 3.0, 6.0])
    selected = select_symmetric_reference(values, 2, which, sigma=1.0)
    np.testing.assert_allclose(np.sort(values[selected]), expected, atol=0, rtol=0)


def test_inverse_both_ends_odd_extra_from_larger_transformed_end():
    values = np.array([-10.0, -1.0, 0.9, 1.1, 3.0, 6.0])
    selected = select_symmetric_reference(values, 3, "BE", sigma=1.0)
    np.testing.assert_array_equal(np.sort(values[selected]), [0.9, 1.1, 3.0])


@pytest.mark.parametrize(
    "key,k,ncv,shift", [("SymEigsShift_2", 3, 8, 1.0), ("SymEigsShift_7", 10, 48, 10.0)]
)
def test_shifted_far_modes_with_explicit_basis_and_changed_starts(key, k, ncv, shift, tmp_path):
    a = np.load(Path(__file__).parent / "fixtures/spectra_shifted_far.npz")[key]
    ref, basis = np.linalg.eigh(a)
    # Independent physical-distance definition; do not reuse report selector.
    selected = np.argsort(-abs(ref - shift))[:k]
    expected = ref[selected]
    allowed = basis[:, selected]
    rng = np.random.default_rng(204)
    starts = wp.array(rng.normal(size=len(a)), dtype=wp.float64)
    inverse = CuDSSInverse(matrix(a - shift * np.eye(len(a))), 1, mtype="symmetric")
    solver = KrylovSchur(inverse, k, ncv=ncv, which="SM", tol=1e-12, initial_vector=starts)
    evaluation = EigenpairEvaluation(SparseOperator(matrix(a)), solver.eigenvectors, tol=1e-9)
    graph = solver.capture(100, finalizer=evaluation.run)
    try:
        for _ in range(2):
            starts.assign(rng.normal(size=len(a)))
            wp.capture_launch(graph)
            values, vectors = evaluation.values.numpy(), solver.eigenvectors.numpy().T
            diagnostics = dict(
                inverse_count=int(solver.converged.numpy()[0]),
                physical_count=int(evaluation.converged.numpy()[0]),
                residual=float(np.max(abs(a @ vectors - vectors * values))),
            )
            np.testing.assert_allclose(
                np.sort(values), np.sort(expected), atol=1e-9, rtol=0, err_msg=str(diagnostics)
            )
            assert np.max(abs(a @ vectors - vectors * values)) < 1e-9
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(k), atol=1e-10, rtol=0)
            assert np.linalg.norm(vectors - allowed @ (allowed.T @ vectors), 2) < 1e-9
            assert solver.converged.numpy()[0] == evaluation.converged.numpy()[0] == k
            assert solver.status.numpy()[0] == 0
        dot = tmp_path / "shifted-far.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        text = dot.read_text()
        assert not re.search(r'label="\{\s*HOST\b', text) and "DtoH" not in text
        # cuDSS HtoD metadata copies remain the separate vendor limitation #29.
    finally:
        solver.graph = graph = None
        inverse.close()


@pytest.mark.parametrize(
    "which,expected",
    [
        ("LM", [-10.0, 6.0]),
        ("SM", [-1.0, 0.9]),
        ("LA", [3.0, 6.0]),
        ("SA", [-10.0, -1.0]),
        ("BE", [-10.0, 6.0]),
    ],
)
def test_reference_selector_preserves_unshifted_hermitian_order(which, expected):
    values = np.array([-10.0, -1.0, 0.9, 1.1, 3.0, 6.0])
    np.testing.assert_array_equal(
        np.sort(values[select_symmetric_reference(values, 2, which)]), expected
    )
