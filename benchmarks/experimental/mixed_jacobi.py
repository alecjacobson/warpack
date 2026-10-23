"""FP32 basis initialization followed by original-matrix FP64 recovery (#160).

Every column is retained through FP32 work. Final eigenvalues, rank decisions,
and acceptance use FP64 data; this benchmark experiment changes no public API.
"""

import warp as wp

from benchmarks.experimental.fp32_global_jacobi import GlobalFP32Jacobi
from benchmarks.experimental.fp32_jacobi import FP32Jacobi
from benchmarks.experimental.fp32_shared_jacobi import SharedFP32Jacobi
from warpack import kernels as K
from warpack._projected_cholesky import global_cholesky
from warpack.block import gram, rotate, symmetrize
from warpack.projected import GlobalJacobi


@wp.kernel(enable_backward=False)
def max_scale(h: wp.array2d[wp.float64], scale: wp.array[wp.float64], status: wp.array[wp.int32]):
    block, lane = wp.tid()
    n = h.shape[0]
    largest = wp.float64(0.0)
    bad = int(0)
    for entry in range(lane, n * n, 256):
        value = wp.abs(h[entry // n, entry % n])
        if not wp.isfinite(value):
            bad = 1
        else:
            largest = wp.max(largest, value)
    largest = wp.tile_extract(wp.tile_max(wp.tile(largest)), 0)
    bad = wp.tile_extract(wp.tile_max(wp.tile(bad)), 0)
    if lane == 0:
        scale[0] = largest
        if largest == wp.float64(0.0):
            scale[0] = wp.float64(1.0)
        if bad != 0:
            status[0] = 1


@wp.kernel(enable_backward=False)
def scaled_copy(
    h: wp.array2d[wp.float64], scale: wp.array[wp.float64], small: wp.array2d[wp.float32]
):
    i, j = wp.tid()
    small[i, j] = wp.float32(h[i, j] / scale[0])


@wp.kernel(enable_backward=False)
def basis_rows(v: wp.array2d[wp.float32], rows: wp.array2d[wp.float64]):
    i, j = wp.tid()
    rows[i, j] = wp.float64(v[j, i])


class MixedJacobi:
    """Same buffer interface as GlobalJacobi, with preallocated recovery work."""

    def __init__(self, h, v, cs, ss, partners, temp, vtemp, *, initializer="persistent"):
        self.n, self.device = h.shape[0], h.device
        if not self.device.is_cuda or not 2 <= self.n <= 512 or self.n % 2:
            raise ValueError("mixed Jacobi requires CUDA and an even dimension in 2..512")
        if initializer not in ("persistent", "shared", "global"):
            raise ValueError("unknown mixed Jacobi initializer")
        self.h, self.v = h, v
        (
            self.original,
            self.basis,
            self.metric,
            self.factor,
            self.factor_work,
            self.image,
            self.lifted,
        ) = (wp.empty((self.n, self.n), dtype=wp.float64, device=self.device) for _ in range(7))
        self.scale = wp.empty(1, dtype=wp.float64, device=self.device)
        self.status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.basis_transpose, self.vector_transpose = self.basis.transpose(), v.transpose()

        def single():
            return wp.empty((self.n, self.n), dtype=wp.float32, device=self.device)

        coarse_type = {
            "persistent": FP32Jacobi,
            "shared": SharedFP32Jacobi,
            "global": GlobalFP32Jacobi,
        }[initializer]
        self.coarse = coarse_type(
            single(),
            single(),
            wp.empty(self.n, dtype=wp.float32, device=self.device),
            wp.empty(self.n, dtype=wp.float32, device=self.device),
            partners,
            single(),
            single(),
        )
        self.polish = GlobalJacobi(h, v, cs, ss, partners, temp, vtemp)
        self.sweeps = self.polish.sweeps

    def apply(self):
        self.status.zero_()
        wp.copy(self.original, self.h)
        wp.launch(symmetrize, self.h.shape, [self.original], device=self.device)
        wp.launch_tiled(
            max_scale,
            dim=1,
            inputs=[self.original, self.scale, self.status],
            block_dim=256,
            device=self.device,
        )
        wp.launch(
            scaled_copy,
            self.h.shape,
            [self.original, self.scale, self.coarse.h],
            device=self.device,
        )
        self.coarse.apply()
        wp.launch(basis_rows, self.h.shape, [self.coarse.v, self.basis], device=self.device)
        # Gram is well conditioned: the complete FP32 basis is near orthogonal.
        # Cholesky QR restores FP64 orthogonality without dropping any column.
        gram(self.basis, self.basis, self.metric, symmetric=True)
        wp.launch_tiled(
            global_cholesky,
            dim=1,
            inputs=[self.metric, self.factor, self.status, self.factor_work],
            block_dim=256 if self.n <= 256 else 512,
            device=self.device,
        )
        wp.launch(K.triangular, self.n, [self.basis, self.factor], device=self.device)
        rotate(self.original, self.basis, self.image)
        gram(self.basis, self.image, self.h, symmetric=True)
        self.polish.apply()
        gram(self.basis_transpose, self.vector_transpose, self.lifted)
        wp.copy(self.v, self.lifted)


class MixedSharedJacobi(MixedJacobi):
    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        super().__init__(h, v, cs, ss, partners, temp, vtemp, initializer="shared")


class MixedGlobalJacobi(MixedJacobi):
    def __init__(self, h, v, cs, ss, partners, temp, vtemp):
        super().__init__(h, v, cs, ss, partners, temp, vtemp, initializer="global")
