"""Every operator used by a generalized solve constrains its outer capture policy."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import GeneralizedKrylovSchur, SparseOperator


class StaticOnly:
    requires_static_outer_capture = True

    def __init__(self, operator):
        self.operator = operator
        self.n, self.device = operator.n, operator.device

    def apply(self, x, y):
        self.operator.apply(x, y)


@pytest.mark.parametrize("mode", ["regular", "shift-invert", "cayley"])
@pytest.mark.parametrize("restricted", ["original", "mass", "inverse"])
def test_generalized_respects_each_operand_and_replays_physical_pairs(mode, restricted):
    n, k, shift = 16, 3, 2.3
    spectrum = np.arange(1.0, n + 1)
    m = np.linspace(1.0, 2.0, n)
    a = spectrum * m
    inv = 1 / (m if mode == "regular" else a - shift * m)
    operators = dict(
        original=SparseOperator(matrix(np.diag(a))),
        mass=SparseOperator(matrix(np.diag(m))),
        inverse=SparseOperator(matrix(np.diag(inv))),
    )
    operators[restricted] = StaticOnly(operators[restricted])
    solver = GeneralizedKrylovSchur(
        operators["original"],
        operators["mass"],
        operators["inverse"],
        k,
        ncv=n,
        target=shift,
        mode=mode,
        which="LA" if mode == "regular" else "LM",
        tol=1e-11,
    )
    assert solver.iter_op.requires_static_outer_capture
    # This must reject before warming or capturing the restricted operation.
    with pytest.raises(ValueError, match="adaptive=False"):
        solver.capture(2, adaptive=True)
    graph = solver.capture(2)
    if mode == "regular":
        expected = spectrum[-k:]
    else:
        transformed = (
            (spectrum + shift) / (spectrum - shift) if mode == "cayley" else 1 / (spectrum - shift)
        )
        expected = spectrum[np.argsort(-abs(transformed))[:k]]
    for _ in range(2):
        wp.capture_launch(graph)
        values, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        np.testing.assert_allclose(np.sort(values), np.sort(expected), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            a[:, None] * vectors, m[:, None] * vectors * values, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(vectors.T @ (m[:, None] * vectors), np.eye(k), atol=1e-11)
        assert solver.converged.numpy()[0] == k
        assert solver.status.numpy()[0] == 0
        assert solver.iterations.numpy()[0] == 2
