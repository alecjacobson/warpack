"""Physical/metric operators also constrain symmetric-iteration capture policy."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import GeneralizedEigensolver, SparseOperator, SymmetricEigensolver


class StaticOnly:
    requires_static_outer_capture = True

    def __init__(self, operator):
        self.operator = operator
        self.n, self.device = operator.n, operator.device

    def apply(self, x, y):
        self.operator.apply(x, y)


@pytest.mark.parametrize(
    "kind,restricted",
    [
        ("symmetric", "original"),
        ("symmetric", "inverse"),
        ("generalized", "original"),
        ("generalized", "mass"),
        ("generalized", "inverse"),
    ],
)
def test_capture_checks_every_used_operand(kind, restricted, monkeypatch):
    n, k = 16, 3
    mass = np.linspace(1, 2, n) if kind == "generalized" else np.ones(n)
    a = np.arange(1.0, n + 1) * mass
    ops = dict(
        original=SparseOperator(matrix(np.diag(a))),
        mass=SparseOperator(matrix(np.diag(mass))),
        inverse=SparseOperator(matrix(np.diag(1 / a))),
    )
    ops[restricted] = StaticOnly(ops[restricted])
    if kind == "generalized":
        solver = GeneralizedEigensolver(
            ops["original"], ops["mass"], ops["inverse"], k, ncv=n, tol=1e-11
        )
    else:
        solver = SymmetricEigensolver(
            ops["original"], k, ncv=n, iteration_operator=ops["inverse"], which="SA", tol=1e-11
        )
    original = solver._prepare_capture

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported adaptive request reached warmup")

    with monkeypatch.context() as patch:
        patch.setattr(solver, "_prepare_capture", forbidden)
        with pytest.raises(ValueError, match="adaptive=False"):
            solver.capture(2, adaptive=True)
    assert solver._prepare_capture == original
    graph = solver.capture(2)
    for _ in range(2):
        wp.capture_launch(graph)
        values, u = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        np.testing.assert_allclose(values, np.arange(1.0, k + 1), atol=1e-10)
        np.testing.assert_allclose(a[:, None] * u, mass[:, None] * u * values, atol=1e-10)
        np.testing.assert_allclose(u.T @ (mass[:, None] * u), np.eye(k), atol=1e-10)
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        assert solver.iterations.numpy()[0] == 2
