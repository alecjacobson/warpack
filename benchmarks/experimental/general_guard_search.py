"""Fixed-budget direct-general search using guards after requested convergence.

Unlike the earlier rejected all-guard search, unfinished requested pairs retain
priority. This changes exploration only; public checkpoints and final original
residual/count checks still decide the returned output. No completeness claim.
"""

import warp as wp

from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.arnoldi import complex_residual, rotate_complex, save_restart_residual
from warpack.schur import cabs


@wp.kernel
def select_guard_after_requested(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    index: wp.array[wp.int32],
    selections: wp.array[wp.int32],
    k: int,
    tol: wp.float64,
):
    ready = bool(True)
    for j in range(k):
        error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
        ready = ready and wp.isfinite(error) and error <= tol
    if ready:
        best = wp.float64(0.0)
        chosen = int(-1)
        for j in range(k, norms.shape[0]):
            error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(values[order[j]]))
            if wp.isfinite(error) and error > best:
                chosen, best = j, error
        if chosen >= 0:
            index[0] = chosen
            selections[0] += 1


class GuardSearchGeneral(StartedGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_images = wp.empty_like(self.retained)
        self.guard_norms = wp.empty((self.keep, 1), dtype=wp.float64, device=self.device)
        self.guard_selections = wp.zeros(1, dtype=wp.int32, device=self.device)

    def initialize(self, seed=42):
        self.guard_selections.zero_()
        super().initialize(seed)

    def prepare_restart(self):
        super().prepare_restart()
        wp.launch(
            rotate_complex,
            self.guard_images.shape,
            [self.z, self.projected, self.order, self.guard_images],
            device=self.device,
        )
        self.guard_norms.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.keep, (self.n + 255) // 256),
            inputs=[self.retained, self.guard_images, self.ritz, self.order, self.guard_norms],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            select_guard_after_requested,
            1,
            [
                self.ritz,
                self.order,
                self.guard_norms,
                self.restart_index,
                self.guard_selections,
                self.k,
                self.tol,
            ],
            device=self.device,
        )
        wp.launch(
            save_restart_residual,
            self.n,
            [
                self.retained,
                self.guard_images,
                self.ritz,
                self.order,
                self.guard_norms,
                self.restart_index,
                self.restart_residual,
            ],
            device=self.device,
        )

    def capture(self, max_iterations=100, adaptive=False, finalizer=None):
        if adaptive is not False:
            raise ValueError("Guard search experiment requires a fixed budget")
        return super().capture(max_iterations, adaptive=False, finalizer=finalizer)
