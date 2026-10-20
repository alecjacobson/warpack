"""Independent direct-general probes after requested residual convergence.

Benchmark-only fixed-budget policy. Each probe replaces the first expansion
vector with a new device-generated direction, projected against retained Ritz
vectors. Public checkpoints protect complete outputs; this is not a global
requested-spectrum certificate or an adaptive stopping implementation.
"""

import warp as wp

from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.arnoldi import complete_restart_direction, restart_direction, scale_vector
from warpack.schur import cabs


@wp.kernel
def requested_ready(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    ready: wp.array[wp.int32],
    cycle: wp.array[wp.int32],
    tol: wp.float64,
):
    complete = bool(True)
    for j in range(norms.shape[0]):
        error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
        complete = complete and wp.isfinite(error) and error <= tol
    ready[0] = int(complete)
    cycle[0] += 1


@wp.kernel
def independent_direction(
    q: wp.array[wp.vec2d],
    ready: wp.array[wp.int32],
    cycle: wp.array[wp.int32],
    used: wp.array[wp.int32],
    recoveries: wp.array[wp.int32],
    probes: wp.array[wp.int32],
):
    i = wp.tid()
    if ready[0] != 0:
        state = wp.rand_init(7919 + 4583 * cycle[0], i)
        q[i] = wp.vec2d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)))
        if i == 0:
            # The candidate prepared by the ordinary restart was overwritten.
            recoveries[0] -= used[0]
            used[0] = 0
            probes[0] += 1


class IndependentSearchGeneral(StartedGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_ready = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.search_cycle = wp.zeros_like(self.requested_ready)
        self.independent_probes = wp.zeros_like(self.requested_ready)

    def initialize(self, seed=42):
        self.requested_ready.zero_()
        self.search_cycle.zero_()
        self.independent_probes.zero_()
        super().initialize(seed)

    def prepare_restart(self):
        super().prepare_restart()
        wp.launch(
            requested_ready,
            1,
            [self.ritz, self.order, self.norms, self.requested_ready, self.search_cycle, self.tol],
            device=self.device,
        )

    def expand(self, j):
        if j != self.keep:
            return super().expand(j)
        # Same expansion as GeneralEigensolver, with one conditional device
        # overwrite before the existing second orthogonalization/normalization.
        self.op.apply_complex(self.views[j - 1], self.zviews[j - 1])
        wp.copy(self.views[j], self.zviews[j - 1])
        self.length(j, self.reference)
        self.orthogonalize(j)
        wp.launch(
            restart_direction,
            self.n,
            [
                self.qrows[j],
                self.norm,
                self.reference,
                self.restart_residual,
                self.restart_index,
                self.residual_restarts,
                self.recovery_used,
                2,
                701 + j,
            ],
            device=self.device,
        )
        wp.launch(
            independent_direction,
            self.n,
            [
                self.qrows[j],
                self.requested_ready,
                self.search_cycle,
                self.recovery_used,
                self.residual_restarts,
                self.independent_probes,
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
            raise ValueError("Independent search experiment requires a fixed budget")
        return super().capture(max_iterations, adaptive=False, finalizer=finalizer)
