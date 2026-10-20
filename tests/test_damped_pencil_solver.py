"""Sparse consistent-mass right modes with direct and iterative device inverses."""

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.damped import quality
from benchmarks.applications.damped_pencil import consistent_chain
from benchmarks.upstream.adapters import complex_matrix, output
from warpack import ComplexShiftInverse
from warpack.pencil import GeneralizedShiftInvertEigensolver


@pytest.mark.parametrize("kind", ["nonproportional", "gyroscopic"])
@pytest.mark.parametrize("backend", ["cudss", "gmres"])
def test_consistent_damped_pencil_replay(kind, backend):
    case = consistent_chain(nodes=12 if backend == "gmres" else 24, k=6, kind=kind)
    shifted = complex_matrix(case.a - case.target * case.b)
    inverse = ComplexShiftInverse(
        shifted,
        backend=backend,
        inner_tol=1e-14,
        inner_maxiter=1000,
        inner_restart=96 if backend == "gmres" else None,
    )
    solver = GeneralizedShiftInvertEigensolver(
        complex_matrix(case.a),
        complex_matrix(case.b),
        inverse,
        case.k,
        sigma=case.target,
        ncv=32,
        tol=1e-10,
    )
    try:
        graph = solver.capture(8)
        for _ in range(2):
            wp.capture_launch(graph)
            check = quality(case, output(solver.eigenvalues), output(solver.eigenvectors).T)
            assert check["accepted"], check
            assert solver.converged.numpy()[0] == case.k
            assert solver.status.numpy()[0] == 0
            assert np.max(solver.residuals.numpy()) <= solver.tol
    finally:
        solver.close()
        graph = None
        inverse.close()
