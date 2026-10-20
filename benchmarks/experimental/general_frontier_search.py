"""Keep an unfinished guard vector while preserving completed requested outputs.

Fixed-budget direct-general experiment. It retains an unconverged extra Ritz
vector as the final restart vector; an independent seed is used if the entire
projected spectrum is resolved. Neither policy certifies unseen eigenvalues.
"""

import warp as wp

from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.arnoldi import (
    complete_restart_direction,
    complex_residual,
    rotate_complex,
    scale_vector,
)
from warpack.schur import cabs


@wp.kernel
def choose_frontier(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    ready: wp.array[wp.int32],
    random_probe: wp.array[wp.int32],
    selections: wp.array[wp.int32],
    k: int,
    keep: int,
    tol: wp.float64,
):
    complete = bool(True)
    for j in range(k):
        error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
        complete = complete and wp.isfinite(error) and error <= tol
    ready[0] = int(complete)
    random_probe[0] = 0
    if complete:
        chosen = int(-1)
        for j in range(k, norms.shape[0]):
            error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
            if chosen < 0 and wp.isfinite(error) and error > tol:
                chosen = j
        if chosen >= 0:
            previous = order[keep - 1]
            order[keep - 1] = order[chosen]
            order[chosen] = previous
            selections[0] += 1
        else:
            random_probe[0] = 1


@wp.kernel
def advance(cycle: wp.array[wp.int32]):
    cycle[0] += 1


@wp.kernel
def frontier_direction(
    q: wp.array[wp.vec2d],
    norm: wp.array[wp.float64],
    reference: wp.array[wp.float64],
    candidate: wp.array[wp.vec2d],
    index: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    used: wp.array[wp.int32],
    ready: wp.array[wp.int32],
    random_probe: wp.array[wp.int32],
    cycle: wp.array[wp.int32],
    probes: wp.array[wp.int32],
    base_seed: int,
):
    i = wp.tid()
    if i == 0:
        used[0] = 0
    if ready[0] != 0:
        if (
            random_probe[0] != 0
            or norm[0] <= wp.float64(1e-26) * reference[0]
            or not wp.isfinite(norm[0])
        ):
            state = wp.rand_init(7919 + 4583 * cycle[0], i)
            q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))
            if i == 0:
                probes[0] += 1
        # Otherwise A times the unfinished final guard is the new chain.
    elif norm[0] <= wp.float64(1e-26) * reference[0] or index[0] >= 0:
        if index[0] >= 0:
            q[i] = candidate[i]
            if i == 0:
                recoveries[0] += 1
                used[0] = 1
        else:
            state = wp.rand_init(base_seed, i)
            q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))


class FrontierSearchGeneral(StartedGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_vectors = wp.empty_like(self.q)
        self.all_images = wp.empty_like(self.q)
        self.all_norms = wp.empty((self.m, 1), dtype=wp.float64, device=self.device)
        self.requested_ready = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.random_probe = wp.zeros_like(self.requested_ready)
        self.search_cycle = wp.zeros_like(self.requested_ready)
        self.frontier_selections = wp.zeros_like(self.requested_ready)
        self.independent_probes = wp.zeros_like(self.requested_ready)

    def initialize(self, seed=42):
        for array in (
            self.requested_ready,
            self.random_probe,
            self.search_cycle,
            self.frontier_selections,
            self.independent_probes,
        ):
            array.zero_()
        super().initialize(seed)

    def select(self):
        super().select()
        for source, output in ((self.q, self.all_vectors), (self.z, self.all_images)):
            wp.launch(
                rotate_complex,
                output.shape,
                [source, self.projected, self.order, output],
                device=self.device,
            )
        self.all_norms.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.m, (self.n + 255) // 256),
            inputs=[self.all_vectors, self.all_images, self.ritz, self.order, self.all_norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            choose_frontier,
            1,
            [
                self.ritz,
                self.order,
                self.all_norms,
                self.requested_ready,
                self.random_probe,
                self.frontier_selections,
                self.k,
                self.keep,
                self.tol,
            ],
            device=self.device,
        )

    def prepare_restart(self):
        super().prepare_restart()
        wp.launch(advance, 1, [self.search_cycle], device=self.device)

    def expand(self, j):
        if j != self.keep:
            return super().expand(j)
        self.op.apply_complex(self.views[j - 1], self.zviews[j - 1])
        wp.copy(self.views[j], self.zviews[j - 1])
        self.length(j, self.reference)
        self.orthogonalize(j)
        wp.launch(
            frontier_direction,
            self.n,
            [
                self.qrows[j],
                self.norm,
                self.reference,
                self.restart_residual,
                self.restart_index,
                self.residual_restarts,
                self.recovery_used,
                self.requested_ready,
                self.random_probe,
                self.search_cycle,
                self.independent_probes,
                701 + j,
            ],
            device=self.device,
        )
        self.orthogonalize(j)
        wp.launch(
            complete_restart_direction,
            self.n,
            [self.qrows[j], self.norm, self.recovery_used, self.residual_fallbacks, 701 + j],
            device=self.device,
        )
        self.orthogonalize(j)
        wp.launch(scale_vector, self.n, [self.qrows[j], self.norm], device=self.device)

    def capture(self, max_iterations=100, adaptive=False, finalizer=None):
        if adaptive is not False:
            raise ValueError("Frontier search experiment requires a fixed budget")
        return super().capture(max_iterations, adaptive=False, finalizer=finalizer)
