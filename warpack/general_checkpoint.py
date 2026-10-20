"""Device-resident preservation of complete converged general Ritz sets.

A checkpoint does not certify an unseen part of the requested spectrum.
Final convergence always uses fresh products with the requested operator.
"""

import warp as wp

from .kernels import identity_order
from .schur import cabs


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
    acceptance_norms: wp.array2d[wp.float64],
    acceptance_values: wp.array[wp.vec2d],
    acceptance_order: wp.array[wp.int32],
    tol: wp.float64,
):
    score = wp.float64(0.0)
    scale = wp.float64(0.0)
    worst = wp.float64(0.0)
    accepted = status[0] == 0
    for j in range(norms.shape[0]):
        value = values[order[j]]
        error = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(value))
        acceptance = wp.sqrt(acceptance_norms[j, 0]) / wp.max(
            wp.float64(1.0), cabs(acceptance_values[acceptance_order[j]])
        )
        accepted = accepted and wp.isfinite(error) and wp.isfinite(acceptance) and acceptance <= tol
        accepted = accepted and wp.isfinite(value[0]) and wp.isfinite(value[1])
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
    output_values: wp.array[wp.vec2d],
):
    j, i = wp.tid()
    if valid[0] != 0:
        vectors[j, i] = saved_vectors[j, i]
    if i == 0:
        value = values[order[j]]
        if valid[0] != 0:
            value = saved_values[j]
        output_values[j] = value


class GeneralCheckpoint:
    """Preallocated state; never modifies the working projected spectrum."""

    def __init__(self, solver):
        self.solver = solver
        self.valid = wp.zeros(1, dtype=wp.int32, device=solver.device)
        self.replace = wp.zeros_like(self.valid)
        self.updates = wp.zeros_like(self.valid)
        self.score = wp.zeros(1, dtype=wp.float64, device=solver.device)
        self.error = wp.zeros_like(self.score)
        self.scale = wp.zeros_like(self.score)
        self.values = wp.empty_like(solver.eigenvalues)
        self.vectors = wp.empty_like(solver.eigenvectors)
        self.output_values = wp.empty_like(self.values)
        self.output_order = wp.empty(solver.k, dtype=wp.int32, device=solver.device)
        wp.launch(identity_order, solver.k, [self.output_order], device=solver.device)

    def reset(self):
        for array in (self.valid, self.replace, self.updates, self.score, self.error, self.scale):
            array.zero_()

    def consider(self):
        s = self.solver
        if s._checkpoint_validator is None:
            norms, values, order, tol = s.norms, s.ritz, s.order, s.tol
        else:
            norms, values, order, tol = s._checkpoint_validator()
        wp.launch(
            choose_checkpoint,
            1,
            [
                s.ritz,
                s.order,
                s.norms,
                s.status,
                self.valid,
                self.replace,
                self.score,
                self.error,
                self.scale,
                self.updates,
                s.which,
                norms,
                values,
                order,
                tol,
            ],
            device=s.device,
        )
        wp.launch(
            save_checkpoint,
            s.eigenvectors.shape,
            [s.eigenvectors, s.ritz, s.order, self.replace, self.vectors, self.values],
            device=s.device,
        )

    def restore(self):
        s = self.solver
        wp.launch(
            restore_checkpoint,
            s.eigenvectors.shape,
            [
                s.eigenvectors,
                s.ritz,
                s.order,
                self.valid,
                self.vectors,
                self.values,
                self.output_values,
            ],
            device=s.device,
        )
        return self.output_values, self.output_order
