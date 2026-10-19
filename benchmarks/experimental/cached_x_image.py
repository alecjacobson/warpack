"""Experimental reuse of A X through the block solver's basis transformations.

Uses the public implementation's private launch/projection hooks so this
prototype does not duplicate its orthogonalization algorithm. It is tied to
this checkout, not an installed API. Every iteration still freshly computes
A X for physical residuals; reuse spans only that iteration's projections and
normalization. No multistep accumulation of operator-image errors is intended.
"""

import weakref

import warp as wp

from warpack import BlockLOBPCG
from warpack import kernels as K
from warpack.block import scale_rows


class _ReuseOperator:
    def __init__(self, owner, original):
        self.owner = weakref.proxy(owner)
        self.original = original
        self.n, self.device = original.n, original.device
        self.requires_static_outer_capture = getattr(
            original, "requires_static_outer_capture", False
        )

    def apply(self, x, y):
        owner = self.owner
        if (
            owner._reuse_image
            and x.ptr == owner.s.ptr
            and x.shape == owner.s.shape
            and y.ptr == owner.az.ptr
            and y.shape == owner.az.shape
        ):
            # The first block was carried through the exact same transforms.
            # New/history search directions still receive a fresh application.
            self.original.apply(x[owner.b :], y[owner.b :])
        else:
            self.original.apply(x, y)


class CachedXBlockLOBPCG(BlockLOBPCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reuse_image = False
        self._original_operator = self.op
        self._image_x = self.az[: self.b]
        self._image_scratch = self.z[: self.b]
        self._image_constraints = (
            None if self.constraints is None else wp.empty_like(self.constraints)
        )
        self.op = _ReuseOperator(self, self.op)

    def initialize(self, seed=42):
        self._reuse_image = False
        super().initialize(seed)
        if self.constraints is not None:
            # Refresh after the base initializer has refreshed/normalized the
            # current invariant basis. Nonzero invariant eigenvalues are valid.
            self._original_operator.apply(self.constraints, self._image_constraints)

    def step(self):
        wp.copy(self._image_x, self.ax)
        self._reuse_image = True
        try:
            super().step()
        finally:
            self._reuse_image = False

    def subtract(self, basis, coeff, target, scratch):
        super().subtract(basis, coeff, target, scratch)
        if self._reuse_image and target.ptr == self.s.ptr and target.shape == self.s.shape:
            # A(X-Q C) = AX-(AQ) C, using the same projection coefficients.
            super().subtract(
                self._image_constraints, coeff[: self.b], self._image_x, self._image_scratch
            )

    def launch(self, kernel, shape, args):
        super().launch(kernel, shape, args)
        if (
            self._reuse_image
            and kernel in (scale_rows, K.triangular)
            and args[0].ptr == self.sx.ptr
            and args[0].shape == self.sx.shape
        ):
            # Mirror both CholeskyQR passes, including the initial row scaling.
            super().launch(kernel, shape, [self._image_x, *args[1:]])
