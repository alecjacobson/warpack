"""Benchmark adapter: complex SPD pencils through public realified operators.

The installed API remains real generalized / complex standard. This adapter
recovers complex directions in the mass metric using only Warp device kernels.
"""

import warp as wp

from warpack import GeneralizedKrylovSchur
from warpack.generalized import generalized_outputs, generalized_residual
from warpack.hermitian import (
    accept_value,
    accept_vector,
    candidate,
    cdot,
    complex_dots,
    complex_subtract,
)


@wp.func
def metric_product(a: wp.vec2d, b: wp.vec2d):
    return cdot(a, b)[0]


@wp.kernel
def metric_norm(q: wp.array[wp.vec2d], mq: wp.array[wp.vec2d], norm: wp.array[wp.float64]):
    chunk = wp.tid()
    a = wp.tile_load(q, shape=256, offset=chunk * 256)
    b = wp.tile_load(mq, shape=256, offset=chunk * 256)
    wp.tile_atomic_add(norm, wp.tile_sum(wp.tile_map(metric_product, a, b)))


@wp.kernel
def pack(v: wp.array2d[wp.vec2d], out: wp.array2d[wp.float64]):
    j, i = wp.tid()
    out[j, i] = v[j, i][0]
    out[j, i + v.shape[1]] = v[j, i][1]


@wp.kernel
def certify(
    count: wp.array[wp.int32],
    status: wp.array[wp.int32],
    residuals: wp.array[wp.float64],
    converged: wp.array[wp.int32],
    tol: wp.float64,
):
    j = wp.tid()
    if j >= count[0] or status[0] != 0:
        residuals[j] = wp.float64(wp.inf)
    elif wp.isfinite(residuals[j]) and residuals[j] <= tol:
        wp.atomic_add(converged, 0, 1)


class ConnectionEigensolver:
    """Mass-weighted duplicate removal; ncv counts COMPLEX search directions.

    Realification uses 2*k output rows and 2*ncv search rows. The supplied
    inverse solves the realification of K-target*M. Factors must be rebuilt
    after coefficient changes. Initial vectors are realified FP64 arrays.
    """

    def __init__(
        self, operator, mass, inverse, k, ncv=32, target=-0.1, tol=1e-10, initial_vector=None
    ):
        self.n, self.k, self.device = operator.complex_n, k, operator.device
        self.operator, self.mass, self.tol = operator, mass, tol
        self.real = GeneralizedKrylovSchur(
            operator,
            mass,
            inverse,
            2 * k,
            ncv=2 * ncv,
            target=target,
            tol=tol,
            initial_vector=initial_vector,
        )
        self.eigenvectors = wp.zeros((k, self.n), dtype=wp.vec2d, device=self.device)
        self.mass_vectors = wp.zeros_like(self.eigenvectors)
        self.eigenvalues = wp.zeros(k, dtype=wp.float64, device=self.device)
        self.residuals = wp.empty_like(self.eigenvalues)
        self.q = wp.empty(self.n, dtype=wp.vec2d, device=self.device)
        self.mq = wp.empty_like(self.q)
        self.dots = wp.zeros((k, 1), dtype=wp.vec2d, device=self.device)
        self.norm = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.converged = wp.zeros_like(self.count)
        self._unchecked = wp.zeros_like(self.count)
        self.packed = wp.empty((k, 2 * self.n), dtype=wp.float64, device=self.device)
        self.ax = wp.empty_like(self.packed)
        self.bx = wp.empty_like(self.packed)
        self.norms = wp.zeros((k, 3), dtype=wp.float64, device=self.device)
        self.order = wp.array(list(range(k)), dtype=wp.int32, device=self.device)

    def finalize(self):
        self.eigenvectors.zero_()
        self.mass_vectors.zero_()
        self.eigenvalues.zero_()
        self.count.zero_()
        for j in range(2 * self.k):
            wp.launch(
                candidate, self.n, [self.real.eigenvectors, self.q, j, self.n], device=self.device
            )
            wp.launch(candidate, self.n, [self.real.bx, self.mq, j, self.n], device=self.device)
            for _ in range(2):
                self.dots.zero_()
                wp.launch_tiled(
                    complex_dots,
                    dim=(self.k, (self.n + 255) // 256),
                    inputs=[self.eigenvectors, self.mq, self.dots],
                    block_dim=128,
                    device=self.device,
                )
                for basis, vector in ((self.eigenvectors, self.q), (self.mass_vectors, self.mq)):
                    wp.launch(
                        complex_subtract,
                        self.n,
                        [basis, vector, self.dots, self.count],
                        device=self.device,
                    )
            self.norm.zero_()
            wp.launch_tiled(
                metric_norm,
                dim=(self.n + 255) // 256,
                inputs=[self.q, self.mq, self.norm],
                block_dim=128,
                device=self.device,
            )
            for vector, basis in ((self.q, self.eigenvectors), (self.mq, self.mass_vectors)):
                wp.launch(
                    accept_vector,
                    self.n,
                    [vector, basis, self.norm, self.count],
                    device=self.device,
                )
            wp.launch(
                accept_value,
                1,
                [self.real.eigenvalues, self.eigenvalues, self.norm, self.count, j],
                device=self.device,
            )
        wp.launch(
            pack, self.eigenvectors.shape, [self.eigenvectors, self.packed], device=self.device
        )
        self.operator.apply(self.packed, self.ax)
        self.mass.apply(self.packed, self.bx)
        self.norms.zero_()
        self._unchecked.zero_()
        self.converged.zero_()
        wp.launch_tiled(
            generalized_residual,
            dim=(self.k, (2 * self.n + 255) // 256),
            inputs=[self.packed, self.ax, self.bx, self.eigenvalues, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            generalized_outputs,
            self.k,
            [
                self.eigenvalues,
                self.order,
                self.norms,
                self.eigenvalues,
                self.residuals,
                self.tol,
                self._unchecked,
            ],
            device=self.device,
        )
        wp.launch(
            certify,
            self.k,
            [self.count, self.real.status, self.residuals, self.converged, self.tol],
            device=self.device,
        )

    def capture(self, steps):
        return self.real.capture(steps, finalizer=self.finalize)
