"""Largest algebraic modes via the public smallest-algebraic block solver."""

import warp as wp
import warp.sparse as ws

from warpack import BlockLOBPCG, SparseOperator


@wp.kernel
def restore_eigenvalue_sign(source: wp.array[wp.float64], output: wp.array[wp.float64]):
    i = wp.tid()
    output[i] = -source[i]


class LargestBlock:
    """Benchmark-only LA adapter; this is not general largest-magnitude selection.

    Negate K's products using the existing SparseOperator sign parameter and
    restore eigenvalue signs on device. The SPD metric and preconditioner stay
    unchanged. Original matrix coefficients and supplied starts remain live
    inputs to captured replay; references are never used in initialization.
    """

    def __init__(self, stiffness, k, *, mass=None, **kwargs):
        self.core = BlockLOBPCG(
            SparseOperator(stiffness, sign=-1.0),
            k,
            mass=SparseOperator(mass) if mass is not None else None,
            **kwargs,
        )
        self.eigenvalues = wp.empty_like(self.core.eigenvalues)
        self.graph = None

    def __getattr__(self, name):
        return getattr(self.core, name)

    def recover(self):
        wp.launch(
            restore_eigenvalue_sign,
            self.k,
            [self.core.eigenvalues, self.eigenvalues],
            device=self.device,
        )

    def capture(self, max_iterations=100):
        self.graph = self.core.capture(max_iterations, finalizer=self.recover)
        return self.graph


def upper_shifted_matrix(stiffness, mass, upper_shift):
    """Construct upper_shift*M-K on device for a caller-supplied upper bound.

    Setup builds a separate sparse topology; sources stay unmodified. Rebuild
    when coefficients change. SPD validity is the caller's responsibility.
    """
    shifted = ws.bsr_copy(mass)
    ws.bsr_axpy(stiffness, shifted, alpha=-1.0, beta=upper_shift)
    shifted.nnz_sync()
    return shifted
