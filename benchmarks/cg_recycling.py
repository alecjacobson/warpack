"""Experimental initial guesses; all algebra and history stay on device."""

import warp as wp
import warp.optim.linear as linear

from warpack import CGInverse, KrylovSchur
from warpack import kernels as K
from warpack.generalized import accumulate_cg_statistics


@wp.kernel
def combine(
    basis: wp.array2d[wp.float64],
    coefficients: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
    count: int,
    add: float,
):
    i = wp.tid()
    value = wp.float64(0.0)
    if add != 0.0:
        value = out[0, i]
    for j in range(count):
        if add == 0.0:
            value += basis[j, i] * coefficients[j, 0]
        else:
            value -= basis[j, i] * coefficients[j, 0]
    out[0, i] = value


@wp.kernel
def save_direction(
    vector: wp.array2d[wp.float64],
    image: wp.array2d[wp.float64],
    energy: wp.array2d[wp.float64],
    original: wp.array2d[wp.float64],
    basis: wp.array2d[wp.float64],
    images: wp.array2d[wp.float64],
    slot: int,
):
    i = wp.tid()
    scale = wp.float64(0.0)
    if energy[0, 0] > wp.float64(1e-12) * original[0, 0] and energy[0, 0] > wp.float64(1e-300):
        scale = wp.float64(1.0) / wp.sqrt(energy[0, 0])
    basis[slot, i] = scale * vector[0, i]
    images[slot, i] = scale * image[0, i]


class WarmCGInverse(CGInverse):
    def __init__(self, matrix, policy, **kwargs):
        super().__init__(matrix, **kwargs)
        self.policy = policy
        self.capacity = int(policy.removeprefix("recycle")) if policy.startswith("recycle") else 1
        self.basis = wp.zeros((self.capacity, self.n), dtype=wp.float64, device=self.device)
        self.images = wp.zeros_like(self.basis)
        self.vector = wp.zeros((1, self.n), dtype=wp.float64, device=self.device)
        self.image = wp.zeros_like(self.vector)
        self.dots = wp.zeros((self.capacity, 1), dtype=wp.float64, device=self.device)
        self.energy = wp.zeros((1, 1), dtype=wp.float64, device=self.device)
        self.original = wp.zeros_like(self.energy)
        self.operator = linear.aslinearoperator(matrix)
        self.used = 0
        self.cursor = 0

    def reset_history(self):
        self.basis.zero_()
        self.images.zero_()
        self.used = self.cursor = 0

    def dot(self, basis, vector, output, count):
        output.zero_()
        wp.launch_tiled(
            K.gram_partial,
            dim=(count, 1, (self.n + 255) // 256),
            inputs=[basis, vector, output],
            block_dim=128,
            device=self.device,
        )

    def apply(self, x, y):
        for j in range(x.shape[0]):
            rhs, answer = x[j : j + 1], y[j : j + 1]
            if self.policy == "previous":
                wp.copy(answer, self.basis[0:1])
            elif self.used:
                self.dot(self.basis, rhs, self.dots, self.used)
                wp.launch(
                    combine,
                    self.n,
                    [self.basis, self.dots, answer, self.used, 0.0],
                    device=self.device,
                )
            else:
                answer.zero_()
            iterations, residual_sq, tolerance_sq = self.state(b=rhs[0], x=answer[0])
            wp.launch(
                accumulate_cg_statistics,
                1,
                [iterations, residual_sq, tolerance_sq, self.stats],
                device=self.device,
            )
            if self.policy == "previous":
                wp.copy(self.basis[0:1], answer)
                continue
            wp.copy(self.vector, answer)
            self.operator.matvec(self.vector[0], self.image[0], self.image[0], 1.0, 0.0)
            self.dot(self.vector, self.image, self.original, 1)
            if self.policy.startswith("recycle") and self.used:
                for _ in range(2):
                    self.dot(self.basis, self.image, self.dots, self.used)
                    wp.launch(
                        combine,
                        self.n,
                        [self.basis, self.dots, self.vector, self.used, 1.0],
                        device=self.device,
                    )
                    wp.launch(
                        combine,
                        self.n,
                        [self.images, self.dots, self.image, self.used, 1.0],
                        device=self.device,
                    )
            self.dot(self.vector, self.image, self.energy, 1)
            wp.launch(
                save_direction,
                self.n,
                [
                    self.vector,
                    self.image,
                    self.energy,
                    self.original,
                    self.basis,
                    self.images,
                    self.cursor,
                ],
                device=self.device,
            )
            self.cursor = (self.cursor + 1) % self.capacity
            self.used = min(self.used + 1, self.capacity)


class FreshKrylovSchur(KrylovSchur):
    def initialize(self, seed=42):
        if hasattr(self.op, "reset_history"):
            self.op.reset_history()
        super().initialize(seed)
