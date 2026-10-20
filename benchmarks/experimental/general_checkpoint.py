"""Experimental fixed-budget preservation of complete converged Ritz sets.

This preserves a previously converged set while allowing further exploration.
It does not certify that unseen requested eigenvalues have been found. The
final original-operator residual evaluation remains mandatory.
"""

import warp as wp

from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.schur import cabs


@wp.func
def target_score(value: wp.vec2d, which: int):
    score = cabs(value)
    if which == 1:
        score = -score
    elif which == 2:
        score = value[0]
    elif which == 3:
        score = -value[0]
    elif which == 4:
        score = value[1]
    elif which == 5:
        score = -value[1]
    return score


@wp.kernel
def choose_checkpoint(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
    valid: wp.array[wp.int32],
    replace: wp.array[wp.int32],
    previous_score: wp.array[wp.float64],
    previous_error: wp.array[wp.float64],
    previous_scale: wp.array[wp.float64],
    updates: wp.array[wp.int32],
    which: int,
    tol: wp.float64,
):
    score = wp.float64(0.0)
    scale = wp.float64(0.0)
    worst = wp.float64(0.0)
    accepted = status[0] == 0
    for j in range(norms.shape[0]):
        value = values[order[j]]
        error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(value))
        accepted = accepted and wp.isfinite(error) and error <= tol
        score += target_score(value, which)
        scale += wp.max(wp.float64(1.0), cabs(value))
        worst = wp.max(worst, error)
    replace[0] = 0
    # A requested tolerance can be much looser than the observed residuals.
    # It must not erase a resolved improvement between accurate clustered modes.
    # This is a comparison heuristic, not a nonnormal eigenvalue-error bound.
    accuracy = wp.max(wp.float64(7.105427357601002e-15), worst + previous_error[0])
    slack = accuracy * wp.max(scale, previous_scale[0])
    if accepted and (
        valid[0] == 0
        or score > previous_score[0] + slack
        or (wp.abs(score - previous_score[0]) <= slack and worst < previous_error[0])
    ):
        replace[0] = 1
        valid[0] = 1
        previous_score[0] = score
        previous_error[0] = worst
        previous_scale[0] = scale
        updates[0] += 1


@wp.kernel
def save_checkpoint(
    vectors: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    replace: wp.array[wp.int32],
    saved_vectors: wp.array2d[wp.vec2d],
    saved_values: wp.array[wp.vec2d],
):
    j, i = wp.tid()
    if replace[0] != 0:
        saved_vectors[j, i] = vectors[j, i]
        if i == 0:
            saved_values[j] = values[order[j]]


@wp.kernel
def restore_checkpoint(
    vectors: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    valid: wp.array[wp.int32],
    saved_vectors: wp.array2d[wp.vec2d],
    saved_values: wp.array[wp.vec2d],
):
    j, i = wp.tid()
    if valid[0] != 0:
        vectors[j, i] = saved_vectors[j, i]
        if i == 0:
            values[order[j]] = saved_values[j]


class CheckpointGeneral(StartedGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_valid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.checkpoint_replace = wp.zeros_like(self.checkpoint_valid)
        self.checkpoint_updates = wp.zeros_like(self.checkpoint_valid)
        self.checkpoint_score = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.checkpoint_error = wp.zeros_like(self.checkpoint_score)
        self.checkpoint_scale = wp.zeros_like(self.checkpoint_score)
        self.checkpoint_values = wp.empty_like(self.eigenvalues)
        self.checkpoint_vectors = wp.empty_like(self.eigenvectors)
        self.working_values = wp.empty_like(self.ritz)

    def initialize(self, seed=42):
        self.checkpoint_valid.zero_()
        self.checkpoint_updates.zero_()
        super().initialize(seed)

    def prepare_restart(self):
        super().prepare_restart()
        wp.launch(
            choose_checkpoint,
            1,
            [
                self.ritz,
                self.order,
                self.norms,
                self.status,
                self.checkpoint_valid,
                self.checkpoint_replace,
                self.checkpoint_score,
                self.checkpoint_error,
                self.checkpoint_scale,
                self.checkpoint_updates,
                self.which,
                self.tol,
            ],
            device=self.device,
        )
        wp.launch(
            save_checkpoint,
            self.eigenvectors.shape,
            [
                self.eigenvectors,
                self.ritz,
                self.order,
                self.checkpoint_replace,
                self.checkpoint_vectors,
                self.checkpoint_values,
            ],
            device=self.device,
        )

    def capture(self, max_iterations=100, adaptive=False, finalizer=None):
        if adaptive is not False:
            raise ValueError("Experimental checkpoints require fixed capture (adaptive=False)")
        return super().capture(max_iterations, adaptive=False, finalizer=finalizer)

    def finalize(self):
        wp.copy(self.working_values, self.ritz)
        wp.launch(
            restore_checkpoint,
            self.eigenvectors.shape,
            [
                self.eigenvectors,
                self.ritz,
                self.order,
                self.checkpoint_valid,
                self.checkpoint_vectors,
                self.checkpoint_values,
            ],
            device=self.device,
        )
        super().finalize()
        # Keep the working Schur spectrum consistent with h/projected. The
        # returned snapshot can precede the final projected search state.
        wp.copy(self.ritz, self.working_values)
