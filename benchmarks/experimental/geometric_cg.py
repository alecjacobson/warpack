"""Experimental geometric preconditioner adapter for public batched CG/buckets.

All CG recurrence, mixed-precision refresh, output conversion and statistics
come from the public adapter and upstream Warp. Only M's application differs.
Hierarchy factors are shared by width variants; workspaces are independent.
"""

import warp as wp
from warp.optim import linear

from warpack import BatchedCGPreconditioner
from warpack.batched_cg import axpby
from warpack.generalized import _cg_preallocates_loop_control


class GeometricCG(BatchedCGPreconditioner):
    def __init__(
        self,
        matrix,
        hierarchy,
        *,
        maxiter=25,
        tol=0.01,
        working_dtype=wp.float32,
        _working_matrix=None,
    ):
        if (
            hierarchy.n != matrix.shape[0]
            or hierarchy.device != matrix.device
            or hierarchy.dtype != working_dtype
        ):
            raise ValueError("hierarchy must match matrix and CG working precision")
        super().__init__(
            matrix,
            hierarchy.width,
            tol=tol,
            atol=0.0,
            maxiter=maxiter,
            working_dtype=working_dtype,
            matrix_layout="columns",
            _working_matrix=_working_matrix,
        )
        self.hierarchy = hierarchy
        self.work = wp.empty((self.width, self.n), dtype=working_dtype, device=self.device)
        self.geometric_operator = linear.LinearOperator(
            (self.width * self.n, self.width * self.n),
            working_dtype,
            self.device,
            self.matvec,
            batch_offsets=self.offsets,
            max_batch_length=self.n,
        )
        # Replace the unpreconditioned state built by the public constructor;
        # no solve has run. Retain its storage/refresh/statistics implementation.
        self.state = linear.cg(
            self.operator.linear,
            self.b,
            self.x,
            M=self.geometric_operator,
            tol=tol,
            atol=0.0,
            maxiter=maxiter,
            check_every=0,
            run=False,
        )
        self.requires_static_outer_capture = not _cg_preallocates_loop_control(self.state)

    def matvec(self, x, y, z, alpha, beta):
        self.hierarchy.apply(x.reshape((self.width, self.n)), self.work)
        wp.launch(
            axpby,
            z.size,
            [self.work.flatten(), y, z, wp.float64(alpha), wp.float64(beta)],
            device=self.device,
        )

    def for_width(self, width):
        return GeometricCG(
            self.source_matrix,
            self.hierarchy.for_width(width),
            maxiter=self.maxiter,
            tol=self.tol,
            working_dtype=self.working_dtype,
            _working_matrix=self.operator.matrix,
        )
