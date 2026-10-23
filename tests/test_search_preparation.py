"""Direct branch recording must preserve eager setup for custom applications."""

import numpy as np
import pytest
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator


class LazySparse(SparseOperator):
    def __init__(self, a):
        super().__init__(a)
        self.prepared_shapes = set()

    def apply(self, x, y):
        if x.shape not in self.prepared_shapes:
            if wp.get_stream(self.device).is_capturing:
                raise RuntimeError("Custom operator requires eager preparation for each shape")
            self.prepared_shapes.add(x.shape)
        return super().apply(x, y)


@pytest.mark.parametrize(
    "custom", ["none", "operator_subclass", "mass_subclass", "operator_instance", "mass_instance"]
)
def test_all_branches_record_and_custom_lazy_shapes_are_warmed(custom):
    n, width, k = 72, 8, 4
    diagonal = np.arange(1.0, n + 1)
    operator_type = LazySparse if custom == "operator_subclass" else SparseOperator
    mass_type = LazySparse if custom == "mass_subclass" else SparseOperator
    op, mass = operator_type(matrix(np.diag(diagonal))), mass_type(matrix(np.eye(n)))
    if custom.endswith("_instance"):
        target = op if custom.startswith("operator") else mass
        original = target.apply
        prepared = set()

        def application(x, y):
            if x.shape not in prepared:
                if wp.get_stream(target.device).is_capturing:
                    raise RuntimeError("Instance override requires eager preparation")
                prepared.add(x.shape)
            return original(x, y)

        target.apply = application
    solver = BlockLOBPCG(
        op,
        k,
        width,
        mass=mass,
        preconditioner=SparseOperator(matrix(np.diag(1.0 / diagonal))),
        reuse_operator_images=True,
        search_bucket_size=2,
        tol=1e-10,
    )
    eager, recorded = [], []
    step = solver._search.step_bucket

    def track(branch, index):
        (recorded if wp.get_stream(solver.device).is_capturing else eager).append(index)
        return step(branch, index)

    solver._search.step_bucket = track
    graph = solver.capture(100)
    assert recorded == list(range(len(solver._search.buckets)))
    # The initial eager solve uses the largest bucket; only the legacy custom
    # preparation also executes the first/smaller bucket before recording.
    assert (0 in eager) == (custom != "none")
    for scale in [1.0, 1.25, 1.0]:
        op.matrix.values.assign(scale * diagonal)
        wp.capture_launch(graph)
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        values, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        np.testing.assert_allclose(values, scale * diagonal[:k], rtol=0, atol=1e-9)
        assert np.linalg.norm(scale * diagonal[:, None] * vectors - vectors * values) < 1e-8


def test_preparation_falls_back_without_conditional_graph_support(monkeypatch):
    n = 48
    op = SparseOperator(matrix(np.diag(np.arange(1.0, n + 1))))
    solver = BlockLOBPCG(op, 3, 8, reuse_operator_images=True, search_bucket_size=2)
    solver.solve(1)
    prepared = []
    original = solver._search.step_bucket

    def step(branch, index):
        prepared.append(index)
        return original(branch, index)

    solver._search.step_bucket = step
    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: False)
    solver._search.warm()
    assert prepared == list(range(len(solver._search.buckets)))
