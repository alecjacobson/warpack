"""Device-checked Ritz-basis reuse on the qualified wide Ada block path.

The bound concerns the computed mass Gram, not exact-arithmetic products.
Uncertain metrics retain both original QR passes. No numerical host decisions.
"""

import warp as wp

from .batched_sparse import BatchedSparseOperator
from .block import Identity
from .solver import SparseOperator


def owned_mass(mass):
    return (
        type(mass) in (SparseOperator, BatchedSparseOperator, Identity)
        and getattr(mass.apply, "__func__", None) is type(mass).apply
    )


def qualified(owner):
    return (
        getattr(owner, "_ritz_qr_enabled", True)
        and owner.device.is_cuda
        and owner.device.arch == 89
        and 96 <= owner.b <= 256
        and owner.gram_chunk == 4096
        and owner.tol >= 1e-11
        and owner.reuse_operator_images
        and owner.symmetric_products
        and owned_mass(owner.mass)
        and wp.is_conditional_graph_supported()
    )


@wp.kernel
def metric_defect(
    gram: wp.array2d[wp.float64],
    bound: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
):
    i = wp.tid()
    row, column = wp.float64(0.0), wp.float64(0.0)
    for j in range(gram.shape[0]):
        diagonal = wp.float64(i == j)
        a, b = gram[i, j] - diagonal, gram[j, i] - diagonal
        if not wp.isfinite(a) or not wp.isfinite(b):
            wp.atomic_max(invalid, 0, 1)
        row += wp.abs(a)
        column += wp.abs(b)
    # Inflate the computed row/column sums against summation roundoff. This
    # bounds the computed metric defect; it is not an error bound for forming
    # the physical matrix product itself.
    inflation = wp.float64(1.0) + wp.float64(8.0 * 2.220446049250313e-16) * wp.float64(
        gram.shape[0]
    )
    wp.atomic_max(bound, 0, wp.max(row, column) * inflation)


@wp.kernel
def choose_qr(
    bound: wp.array[wp.float64],
    invalid: wp.array[wp.int32],
    status: wp.array[wp.int32],
    fallback: wp.array[wp.int32],
    usage: wp.array[wp.int32],
    tolerance: wp.float64,
):
    reuse = invalid[0] == 0 and status[0] == 0 and wp.isfinite(bound[0]) and bound[0] <= tolerance
    fallback[0] = int(not reuse)
    if reuse:
        usage[0] += 1
    else:
        usage[1] += 1


class MetricQRReuse:
    def __init__(self, owner, storage, tolerance):
        self.owner, self.tolerance = owner, tolerance
        self.bound, self.invalid, self.fallback, self.usage = storage
        mass = owner.mass
        self.eligible = (
            owner.device.is_cuda and wp.is_conditional_graph_supported() and owned_mass(mass)
        )

    def full_qr(self):
        self.owner._orthogonalize_x_full()

    def apply(self):
        owner = self.owner
        if not self.eligible or not wp.get_stream(owner.device).is_capturing:
            self.full_qr()
            return
        owner.mass.apply(owner.sx, owner.bx)
        owner.gram(owner.sx, owner.bx, owner.gx, symmetric=True)
        self.bound.zero_()
        self.invalid.zero_()
        wp.launch(metric_defect, owner.b, [owner.gx, self.bound, self.invalid], device=owner.device)
        wp.launch(
            choose_qr,
            1,
            [self.bound, self.invalid, owner.status, self.fallback, self.usage, self.tolerance],
            device=owner.device,
        )
        wp.capture_if(self.fallback, self.full_qr)
