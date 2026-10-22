"""#144: compose public BlockLOBPCG for dominant nonmass-metric problems.

This is an application experiment, not a new eigensolver or a new CG method.
Minimize the spectrum of -A in metric B, then reverse eigenvalue signs.
"""

import numpy as np
import warp as wp
from warp.optim import linear

from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG


@wp.kernel
def positive_values(negative: wp.array[wp.float64], positive: wp.array[wp.float64]):
    i = wp.tid()
    positive[i] = -negative[i]


class SurfaceBlockSolver:
    def __init__(self, p, width=None, *, inner_steps=25, tol=1e-11):
        width = max(4, 2 * p.k) if width is None else width
        self.a, self.b = matrix(-p.a), matrix(p.b)
        self.factor = linear.FSAI(self.b, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
        pre = BatchedCGPreconditioner(
            self.b,
            width,
            preconditioner=self.factor,
            tol=0.01,
            atol=0.0,
            maxiter=inner_steps,
            working_dtype=wp.float32,
            fsai_layout="columns",
            matrix_layout="columns",
        )
        self.starts = wp.array(
            np.random.default_rng(136).normal(size=(width, p.n)), dtype=wp.float64
        )
        self.solver = BlockLOBPCG(
            BatchedSparseOperator(self.a),
            p.k,
            width,
            mass=BatchedSparseOperator(self.b),
            preconditioner=pre,
            initial_vectors=self.starts,
            tol=tol,
            reuse_operator_images=True,
            search_bucket_size=4,
            keep_history=True,
            preconditioner_bucket_size=4,
            symmetric_products=True,
            fused_updates=True,
            paired_rotation=True,
        )
        self.graph = None
        self.values = wp.empty(p.k, dtype=wp.float64, device=self.a.device)

    def finalize(self):
        wp.launch(
            positive_values,
            self.values.size,
            [self.solver.eigenvalues, self.values],
            device=self.a.device,
        )

    def capture(self, steps=200):
        self.graph = self.solver.capture(steps, finalizer=self.finalize)
        return self.graph

    def solve(self):
        wp.capture_launch(self.graph)

    def output(self):
        return self.values.numpy(), self.solver.eigenvectors.numpy().T

    def close(self):
        pass
