"""Setup-only instrumentation of a general solver's captured device iteration."""

import warp as wp

import warpack.arnoldi as arnoldi


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
    temporarily intercepts launches and must run in its own benchmark process.
    """
    core = getattr(solver, "core", solver)
    trace = wp.full((budget, core.k), float("nan"), dtype=wp.float64, device=core.device)
    h, q = wp.empty_like(core.h), wp.empty_like(core.q)
    original_finalize, original_launch = core.finalize, wp.launch

    def finalize():
        original_finalize()
        wp.launch(
            record_trace, core.k, [core.residuals, core.iterations, trace], device=core.device
        )

    def launch(kernel, *args, **kwargs):
        if kernel is arnoldi.schur:
            wp.copy(h, core.h)
            wp.copy(q, core.q)
        return original_launch(kernel, *args, **kwargs)

    try:
        core.finalize, wp.launch = finalize, launch
        graph = solver.capture(budget)
    finally:
        core.finalize, wp.launch = original_finalize, original_launch
    return graph, (trace, h, q)
