"""Block conformal adapter with truncated upstream batched CG preconditioning.

This is benchmark composition of the public block solver, not a core solver or
an exact inverse. Final acceptance always checks the full original pencil.
"""

import numpy as np
import warp as wp
import warp.optim.linear as linear

from benchmarks.applications.conformal_warp import ConformalEigensolver, scale_reciprocals
from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG

from .batched_cg import BatchedCGInverse
from .boundary_projector import CenteredBoundary, ShiftedConformal


class NegativeBoundary:
    """sigma*M has smallest values gamma=sigma/(lambda-sigma) in [-1,0)."""

    def __init__(self, case, capacity):
        self.boundary = CenteredBoundary(case, capacity)
        self.n, self.device, self.scale = case.n, self.boundary.device, case.shift

    def apply(self, x, y):
        self.boundary.apply(x, y)
        wp.launch(
            scale_reciprocals, y.size, [y.flatten(), self.scale, y.flatten()], device=self.device
        )


class ConformalBlockEigensolver(ConformalEigensolver):
    truncated_preconditioner = True

    def __init__(self, case, width=24, inner_steps=25, outer_tol=1e-12, working_dtype=wp.float32):
        if not hasattr(linear, "FSAI") or not hasattr(linear.LinearOperator, "batch_offsets"):
            raise RuntimeError(
                "conformal block preconditioning requires pinned upstream FSAI/batched CG"
            )
        if not case.k <= width or 3 * width > case.n or inner_steps < 1:
            raise ValueError("require k<=width, 3*width<=n, and positive inner steps")
        stiffness, mass = case.reduced
        shifted = matrix(stiffness - case.shift * mass)
        preconditioner = linear.FSAI(
            shifted, max_row_size=32, max_step_size=3, apply_lanes=8, factor_dtype=wp.float32
        )
        inverse = BatchedCGInverse(
            shifted,
            width,
            tol=0.01,
            atol=0.0,
            maxiter=inner_steps,
            preconditioner=preconditioner,
            working_dtype=working_dtype,
        )
        start = wp.array(
            np.random.default_rng(2026).normal(size=(width, case.n)),
            dtype=wp.float64,
            device=shifted.device,
        )
        core = BlockLOBPCG(
            NegativeBoundary(case, 3 * width),
            case.k,
            width,
            mass=ShiftedConformal(case, 3 * width),
            preconditioner=inverse,
            initial_vectors=start,
            tol=outer_tol,
            gram_chunk=4096,
            symmetric_products=True,
            fused_updates=True,
            paired_rotation=True,
        )
        # core gamma = sigma*mu. Preserve its negative values and strict count;
        # recover positive mu in separate device storage before finalization.
        self._allocate_outputs(case, core, inverse, 1e-10, mu_scale=1 / case.shift)
