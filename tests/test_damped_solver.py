"""Captured right modes of coupled damped FEM systems, in original equations."""

import numpy as np
import pytest
import warp as wp

from benchmarks.applications.damped import coupled_chain, quality
from benchmarks.applications.damped_backends import warp_setup
from benchmarks.upstream.adapters import output


@pytest.mark.parametrize("kind", ["proportional", "nonproportional", "gyroscopic", "coupled-mass"])
@pytest.mark.parametrize("balance", [1.0, 25.0])
def test_damped_cudss_modes_and_replay(kind, balance):
    case = coupled_chain(nodes=24, k=8, kind=kind)
    solver = warp_setup(case, 32, balance=balance)
    try:
        graph = solver.capture(20)
        for _ in range(2):
            wp.capture_launch(graph)
            values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
            assert quality(case, values, vectors)["accepted"]
            assert solver.converged.numpy()[0] == case.k
            assert solver.status.numpy()[0] == 0
    finally:
        solver.close()


def test_damped_gmres_modes():
    case = coupled_chain(nodes=8, k=4, kind="gyroscopic")
    solver = warp_setup(case, 24, backend="gmres")
    try:
        graph = solver.capture(12)
        wp.capture_launch(graph)
        values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
        assert quality(case, values, vectors)["accepted"]
        assert solver.converged.numpy()[0] == case.k
        assert np.isfinite(solver.residuals.numpy()).all()
    finally:
        solver.close()


@pytest.mark.parametrize("balance", [1.0, 25.0])
@pytest.mark.parametrize("kinematic", [False, True])
def test_inverse_polish_preserves_original_modes(balance, kinematic):
    case = coupled_chain(nodes=24, k=8, kind="nonproportional")
    solver = warp_setup(case, 32, balance=balance, polish=1, kinematic=kinematic)
    try:
        graph = solver.capture(3)
        for _ in range(2):
            wp.capture_launch(graph)
            values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
            assert quality(case, values, vectors)["accepted"]
            assert solver.converged.numpy()[0] == case.k
            assert solver.status.numpy()[0] == 0
    finally:
        solver.close()
