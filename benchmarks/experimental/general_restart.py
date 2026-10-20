"""Diagnostic residual recovery for a stalled nonsymmetric restart."""

import warp as wp

from warpack.arnoldi import (
    GeneralEigensolver,
    complex_residual,
    rotate_complex,
    scale_vector,
)
from warpack.hermitian import cmul
from warpack.schur import cabs


@wp.kernel
def worst_residual(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    index: wp.array[wp.int32],
):
    # Search selection is not a stopping test. In shift-invert, a small
    # transformed residual can still leave the original equation unfinished.
    best = wp.float64(0.0)
    index[0] = -1
    for j in range(norms.shape[0]):
        score = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
        if wp.isfinite(score) and score > best:
            best = score
            index[0] = j


@wp.kernel
def save_residual(
    x: wp.array2d[wp.vec2d],
    ax: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    index: wp.array[wp.int32],
    out: wp.array[wp.vec2d],
):
    i = wp.tid()
    j = index[0]
    if j >= 0:
        out[i] = (ax[j, i] - cmul(values[order[j]], x[j, i])) / wp.sqrt(norms[j, 0])


@wp.kernel
def recover(
    q: wp.array[wp.vec2d],
    norm: wp.array[wp.float64],
    reference: wp.array[wp.float64],
    candidate: wp.array[wp.vec2d],
    index: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    used: wp.array[wp.int32],
    use_candidate: int,
    seed: int,
):
    i = wp.tid()
    if i == 0:
        used[0] = 0
    if norm[0] <= wp.float64(1.0e-26) * reference[0] or (use_candidate == 2 and index[0] >= 0):
        if use_candidate != 0 and index[0] >= 0:
            q[i] = candidate[i]
            if i == 0:
                recoveries[0] += 1
                used[0] = 1
        else:
            state = wp.rand_init(seed, i)
            q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))


@wp.kernel
def finish_recovery(
    q: wp.array[wp.vec2d],
    norm: wp.array[wp.float64],
    used: wp.array[wp.int32],
    fallbacks: wp.array[wp.int32],
    seed: int,
):
    i = wp.tid()
    # The candidate was normalized before projection. If it is already in the
    # growing basis, retain random completion rather than install a zero row.
    if used[0] != 0 and (norm[0] <= wp.float64(1.0e-26) or not wp.isfinite(norm[0])):
        state = wp.rand_init(seed, i)
        q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))
        if i == 0:
            fallbacks[0] += 1


class ResidualRestartGeneral(GeneralEigensolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restart_residual = wp.empty(self.n, dtype=wp.vec2d, device=self.device)
        self.restart_index = wp.full(1, -1, dtype=wp.int32, device=self.device)
        self.recoveries = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.recovery_used = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.residual_fallbacks = wp.zeros(1, dtype=wp.int32, device=self.device)

    def initialize(self, seed=42):
        self.restart_index.fill_(-1)
        self.recoveries.zero_()
        self.residual_fallbacks.zero_()
        super().initialize(seed)

    def expand(self, j):
        self.op.apply_complex(self.views[j - 1], self.zviews[j - 1])
        wp.copy(self.views[j], self.zviews[j - 1])
        self.length(j, self.reference)
        self.orthogonalize(j)
        wp.launch(
            recover,
            self.n,
            [
                self.qrows[j],
                self.norm,
                self.reference,
                self.restart_residual,
                self.restart_index,
                self.recoveries,
                self.recovery_used,
                2 if j == self.keep else int(j > self.keep),
                701 + j,
            ],
            device=self.device,
        )
        self.orthogonalize(j)
        if j >= self.keep:
            wp.launch(
                finish_recovery,
                self.n,
                [self.qrows[j], self.norm, self.recovery_used, self.residual_fallbacks, 701 + j],
                device=self.device,
            )
            self.orthogonalize(j)
        wp.launch(scale_vector, self.n, [self.qrows[j], self.norm], device=self.device)

    def step(self):
        super().step()
        # z and projected still describe the pre-restart basis. Their rotation
        # supplies a search direction without another operator/inverse call.
        wp.launch(
            rotate_complex,
            self.ax.shape,
            [self.z, self.projected, self.order, self.ax],
            device=self.device,
        )
        self.norms.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.k, (self.n + 255) // 256),
            inputs=[self.eigenvectors, self.ax, self.ritz, self.order, self.norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            worst_residual,
            1,
            [self.ritz, self.order, self.norms, self.restart_index],
            device=self.device,
        )
        wp.launch(
            save_residual,
            self.n,
            [
                self.eigenvectors,
                self.ax,
                self.ritz,
                self.order,
                self.norms,
                self.restart_index,
                self.restart_residual,
            ],
            device=self.device,
        )
