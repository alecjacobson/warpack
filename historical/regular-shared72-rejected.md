# Rejected shared-memory dispatch for the general 72-column solver

Investigation [#82](https://github.com/alecjacobson/warpack/issues/82), related to
[#31](https://github.com/alecjacobson/warpack/issues/31). The existing block solver
has a shared-memory Jacobi path at 72 columns on devices with at least 98,304
bytes of shared memory per block. Reusing it in the general Krylov dispatch
looked promising, but a matched experiment rejected the change.

The [candidate patch](regular-shared72-rejected.patch) changes only dispatch
and consolidates the block override. All projected convergence safeguards and
sweep limits remain. It was tested on the L40 with the pinned Warp 1.18 overlay,
and then fully reverted from production source. This patch is evidence for the
rejected experiment, not a recommended change.

For the 27,648-DOF, 20-mode negative-foundation LM problem, explicit diagonal
mass scaling, ncv72 and the same 20-cycle budget, every numerical gate passes
before and after. The three-trial median changes from **190.695 to 250.498 ms**
(**1.314x slower**), exceeding the unchanged 1.25 regression limit. The CuPy
control changes from 132.591 to 136.062 ms. Both runs are uncontended; the
candidate batch exits nonzero because its performance gate fails.

[Before](../results/performance_followup/negative-lm-scaled72-before.json),
[candidate](../results/performance_followup/negative-lm-scaled72-after.json),
with occupancy companions beside each report. The candidate still completes
all 20 outer iterations; different rounding produces one or two breakdown
recoveries, so the complete run includes that numerical-path difference.

An isolated projected solve restores the same dense PSD matrix before every
sample and validates the complete spectrum, original equation and orthogonality.
It takes **9.890 ms** with shared memory versus **5.334 ms** with the existing
global-memory path (13 sweeps), three batches of ten solves each:
[shared](../results/performance_followup/regular-projected72-shared.json),
[global](../results/performance_followup/regular-projected72-global.json).
This supports retaining the general solver's current dispatch; it does not
establish which path is fastest for every projected matrix or for Anvil's block
iterations. The block solver's existing dispatch is unchanged by this work.

Seventy-one focused candidate tests passed, including difficult complete
projected spectra, captured replay, capability fallbacks and diagonal-mass
recovery. Accuracy alone did not justify promoting a slower path. Final
validation is performed again after reverting; the separate diagonal-mass
benchmark changes remain, with no production numerical source changes.
