"""Device-resident projection onto the complement of a known subspace."""

import warp as wp

from . import kernels as K


@wp.kernel
def subtract_projection(
    basis: wp.array2d[wp.float64],
    coefficients: wp.array2d[wp.float64],
    vector: wp.array2d[wp.float64],
):
    i = wp.tid()
    value = vector[0, i]
    for j in range(basis.shape[0]):
        value -= basis[j, i] * coefficients[j, 0]
    vector[0, i] = value


class OrthogonalComplementOperator:
    """Apply ``P operator P``, where ``P = I - basis.T @ basis``.

    ``basis`` contains orthonormal FP64 rows on the operator device. Supplying a
    known nullspace lets an inverse eigensolver target nonzero modes directly.
    Orthonormality is a caller precondition. Application uses preallocated Warp
    buffers and is graph capturable; no numerical data is read on the host.

    The wrapped operator must accept one RHS at a time. Instances own scratch
    and must not be applied concurrently on independent streams.
    """

    def __init__(self, operator, basis):
        self.operator = operator
        self.n, self.device = operator.n, operator.device
        if basis.ndim != 2 or basis.shape[1] != self.n or basis.shape[0] < 1:
            raise ValueError("basis must have shape (rank, n), with positive rank")
        if basis.dtype != wp.float64 or basis.device != self.device:
            raise ValueError("basis must be FP64 on the operator device")
        self.basis = basis
        self.requires_static_outer_capture = getattr(
            operator, "requires_static_outer_capture", False
        )
        self.rhs = wp.empty((1, self.n), dtype=wp.float64, device=self.device)
        self.dots = wp.zeros((basis.shape[0], 1), dtype=wp.float64, device=self.device)

    def reset_history(self):
        """Forward an optional inverse-history reset; numerical buffers stay on device."""
        if hasattr(self.operator, "reset_history"):
            self.operator.reset_history()

    def _project(self, vector):
        self.dots.zero_()
        wp.launch_tiled(
            K.gram_partial,
            dim=(self.basis.shape[0], 1, (self.n + 255) // 256),
            inputs=[self.basis, vector, self.dots],
            block_dim=128,
            device=self.device,
        )
        wp.launch(subtract_projection, self.n, [self.basis, self.dots, vector], device=self.device)

    def apply(self, x, y):
        for j in range(x.shape[0]):
            wp.copy(self.rhs, x[j : j + 1])
            self._project(self.rhs)
            self.operator.apply(self.rhs, y[j : j + 1])
            self._project(y[j : j + 1])
