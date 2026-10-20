"""Setup-only instrumentation of a general solver's captured device iteration."""

import warp as wp


@wp.kernel
def record_trace(
    errors: wp.array[wp.float64],
    iteration: wp.array[wp.int32],
    out: wp.array2d[wp.float64],
):
    j = wp.tid()
    i = iteration[0]
    if i < out.shape[0]:
        out[i, j] = errors[j]


def capture_trace(solver, budget):
    """Copy the last pre-Schur projection and optional per-cycle residuals.

    The hooks run only during warmup/capture, never as replay callbacks. A
    fixed outer graph only finalizes its core once, so its trace has one row;
    the projected snapshots still record the last iteration. This diagnostic
    temporarily wraps the core projection/finalizer methods during setup.
    """
    core = getattr(solver, "core", solver)
    trace = wp.full((budget, core.k), float("nan"), dtype=wp.float64, device=core.device)
    h, q = wp.empty_like(core.h), wp.empty_like(core.q)
    original_finalize, original_diagonalize = core.finalize, core.diagonalize

    def finalize():
        original_finalize()
        wp.launch(
            record_trace, core.k, [core.residuals, core.iterations, trace], device=core.device
        )

    def diagonalize():
        wp.copy(h, core.h)
        wp.copy(q, core.q)
        return original_diagonalize()

    try:
        core.finalize, core.diagonalize = finalize, diagonalize
        graph = solver.capture(budget)
    finally:
        core.finalize, core.diagonalize = original_finalize, original_diagonalize
    return graph, (trace, h, q)
