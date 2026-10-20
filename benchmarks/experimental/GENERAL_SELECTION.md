# General spectrum-selection diagnostics

The public general solver's original Spectra benchmark profile passes 11/20
requested-set checks. Doubling the explicitly requested `ncv` improves this to
19/20 across three reset graph replays per case. This is an accuracy profile,
not a solver fix or a performance result. The original fixtures, starts, k=10,
tol=1e-10, and maximum 300 device-controlled restart cycles are unchanged.
Widths become 40 for n=100 and 60 for n=1,000. The public retained width remains
14. Production defaults and historical timings are unchanged.

[Full wide-profile results](../../results/general-selection-wide.json) include
all 60 replay validations, values, source/runtime fingerprints, input hashes,
device counts, and executed cycles. Accepted replays check original residuals,
complete target assignment, required eigenvalues above boundary ties, normalized
independent vectors, and the allowed invariant subspace. These diagnostics do
not time execution.

`benchmark_gen_38` remains rejected in all three wide-profile replays: each
stops after 13 cycles with count 10 and a normalized residual below 1e-10, but
returns the wrong requested LM set. The eigenvalue assignment error is about
31.47. The [native reference audit](../upstream/GENERAL_SELECTION.md) also rejects
real Spectra and ARPACK for this case at width 60; complex ARPACK passes. All
native variants pass with the separately labeled width-120 profile. This does
not establish that an arbitrary problem's requested set can be certified from
local residuals.

The source change extracts the existing `sort_complex` launch into `select()`
so diagnostic subclasses can inspect or alter ordering without intercepting
Warp's global launch function. The default selection arithmetic is unchanged.
Exploratory guard and real-basis restart variants are not promoted: recovering
some target sets did not reliably meet both spectrum and residual gates. The
public algorithm, independent-reference coverage, and performance follow-up
remain tracked in [#104](https://github.com/alecjacobson/warpack/issues/104) and
[#14](https://github.com/alecjacobson/warpack/issues/14).

Reproduce using the pinned Warp environment and exported original fixtures:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH="$PWD/build/warp-fsai:$PWD" \
  python -m benchmarks.experimental.general_selection_profiles \
  --ncv-multiplier 2 --budget 300 --repeats 3 \
  --out build/general-selection-wide.json
```

Use `--fixtures` for an existing export and repeated `--case` to select a
subset. Capture warmup precedes the reset replays; original v0 is read again
inside every replay. No additional vendor eigensolver runs inside Warp.

## Longer fixed-budget control

A separate width-60 / fixed-80-cycle control on `benchmark_gen_38` initially
passes all three replays: requested eigenvalue errors below 3e-13 and original
normalized residuals below 8e-15. However, the full 20-case fixed-80 audit passes
only **11/20** cases across three reset replays each. All ten n=100 matrices
pass, but only `benchmark_gen_36` passes all three large-matrix replays. The
other nine large cases each have at least one failed replay. This includes a
later failure for case 38: the isolated successful control was not robust.

Some fixed replay failures have one or two inaccurate requested vectors while
the others remain near machine precision. For example, the third case-30 replay
reports 9/10 with normalized residual 0.09715 and spectrum error 32.54, whereas
its first two fixed replays and all three adaptive wide-profile replays pass.
The comparisons establish trajectory sensitivity and poor final results after
more work; a per-cycle trace is still needed to establish exactly when a given
failed replay had recovered or lost its requested set. Neither a larger fixed
budget nor the isolated successful control is recommended as a general fix.

[Complete fixed audit](../../results/general-selection-wide-fixed80.json),
[isolated case-38 control](../../results/general-selection38-wide-fixed80.json).
All failures remain rejected; no timings are claimed. Investigation and a
partial-space replay regression are tracked in
[#108](https://github.com/alecjacobson/warpack/issues/108).

To reproduce the fixed audit, add `--fixed --budget 80` to the command above.
Use `--ncv N` to select an explicit width instead of the multiplier. These
options alter the reported experimental profile, not production defaults.

## Per-cycle diagnosis

A subsequent instrumented case-30 trace identifies the lost-output mechanism.
All five tiled-Schur replays first pass the complete requested spectrum and
residual check at cycle 23. Each then fails at later cycles (11–17 times per
replay), although all finish cycle 80 successfully in this instrumented run.
The complete desired set remains among the converged Ritz pairs at every such
failed cycle. The orthonormal basis is intact (error below 1.5e-15), and the
projected eigensolve remains accurate (backward error below 1.1e-15).

At cycle 25, the inaccurate estimate `17.6858757 - 4.5601808j`, with normalized
original residual 0.00822, outranks the already converged boundary eigenvalue
`17.6717832 + 4.5666816j`, with residual 3.8e-13. The accurate vector is still
in the projected basis; it has moved just outside the selected top ten.
At later cycles, other unconverged estimates similarly displace converged
requested pairs. The problem in these traces is output ranking during continued
exploration, rather than loss of the underlying invariant directions.

The matched serial-Schur control reproduces this behavior in both 50-cycle
replays: first pass at 23, then nine failed cycles each, always with the complete
converged set still present in the projection. This rules out the tiled Schur
optimization as the necessary cause of this failure.

[Tiled trace](../../results/general-selection-trace30-tiled.json),
[serial trace](../../results/general-selection-trace30-serial.json), and compact
raw projected spectra/residual arrays for
[tiled](../../results/general-selection-trace30-tiled.npz) and
[serial](../../results/general-selection-trace30-serial.npz). Per-cycle gates
check spectrum, residuals and projected status; final output gates additionally
check full vectors/subspaces and numerical independence. Instrumentation adds
kernels and can change roundoff trajectories. These traces diagnose the
mechanism, not the exact trajectory of an earlier uninstrumented failure, and
provide no performance timings.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH="$PWD/build/warp-fsai:$PWD" \
  python -m benchmarks.experimental.general_selection_trace \
  --case benchmark_gen_30 --ncv 60 --budget 80 --repeats 5 \
  --out build/general-selection-trace30-tiled.json
# Serial control: --backend serial --budget 50 --repeats 2
```

Preserving a complete converged result while continuing exploration is being
investigated separately. It must not hide missing requested modes, accept an
inaccurate original equation, or reuse stale results after a changed-matrix
replay. No such policy has yet been promoted to the public solver.

## Experimental preservation of converged results

The benchmark-only `CheckpointGeneral` preserves a complete converged Ritz set
while the fixed-budget search continues unchanged. It compares the sum of the
requested selection scores and, within a tolerance-scaled tie, prefers the
smaller maximum residual. It does not simply freeze the first converged set:
later discoveries can improve the recorded spectrum. The final output receives
the usual fresh original-operator residual evaluation and strict count gate.
The checkpoint decision and conditional vector copies run in Warp on device.

At the same explicit widths (40/60), fixed 80 cycles, k=10, tol=1e-10, original
matrices and original starts, the prototype passes **20/20 cases and all 60 reset
replays**. The unmodified fixed-80 control passes only 11/20 cases across every
replay. Worst accepted normalized original residual is 7.21e-15, requested
spectrum error 2.95e-13, and invariant-subspace error 7.59e-14.
[Full checkpoint results](../../results/general-selection-wide-checkpoint80.json).

Seven initial regressions pass (2.23 s), covering all six selectors, changed
matrix replay, and a previously converged problem followed by an unconverged
one. These prevent stale results from substituting for a new solve. The subsequent small partial-space regression and overhead checks are recorded below. No checkpoint policy is part of
the public solver yet. In particular, the benchmark-only fixed direct-operator
results do not establish accuracy for transformed/physical cuDSS stopping
criteria or imply a global spectral-completeness certificate.

Reproduce by adding `--checkpoint --fixed --budget 80` to the profile command.
This compares labeled algorithm configurations; it does not change the original
upstream fixture settings or certify the original-width adaptive profile. The
prototype uses cached operator images for checkpoint eligibility, followed by
fresh original residuals at return, and needs one additional k-by-n complex
vector buffer. No speedup is claimed before its cost is measured.

## Checkpoint regression and overhead control

A deterministic n=96, k=4, ncv=24 real Gaussian problem (NumPy RNG seed 1)
provides a small partial-space regression without storing a large binary.
The diagnostic first passed at cycle 15, then lost the selected set at ten
later cycles, including 60. The checkpoint version passes three reset
fixed-60 replays with the complete requested set, original residuals and no
host transfers/callbacks. Returning a saved result also preserves the separate
working Schur spectrum instead of overwriting it. The experimental capture API
now explicitly requires fixed iteration.

All **eight** checkpoint regressions pass on pinned Warp (5.12 s) and stock
Warp 1.15 (6.33 s). This is focused prototype validation, not a full public-solver
suite. [Validation record](../../results/performance_followup/general-checkpoint-validation.json).

A matched overhead control uses `benchmark_gen_36`, n=1,000, k=10, ncv=60,
fixed 80 cycles, tol=1e-10, and identical matrix/start buffers. Three synchronized
warm samples per variant alternate order; every timed replay passes all common
accuracy/count gates. Median full-solve time is **4.420655 s** without snapshots
and **4.427092 s** with snapshots (ratio 1.00146, about 0.15% overhead). This is
within ordinary unlocked-clock variation at this sample count, not a speedup
claim or a general overhead bound. Failed fixed-budget cases are not used as
accepted performance baselines.

The checkpoint graph adds 161 kernels, two memsets and two DtoD copies to this
80-cycle solve. Both graphs contain only device-to-device copies and no HOST
nodes. Setup/capture and per-trial validation are excluded from solve timing
and reported separately. The occupancy monitor observed no competing GPU jobs;
its approximately one-second samples are evidence, not an exclusive reservation.
[Timing/accuracy record](../../results/general-checkpoint-overhead.json),
[occupancy](../../results/general-checkpoint-overhead.occupancy.json).

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH="$PWD/build/warp-fsai:$PWD" \
  python -m benchmarks.monitor_gpu \
  --out build/general-checkpoint-overhead.occupancy.json \
  --log build/general-checkpoint-overhead.log -- \
  python -m benchmarks.experimental.compare_general_checkpoint \
  --case benchmark_gen_36 --ncv 60 --budget 80 --repeats 3 \
  --out build/general-checkpoint-overhead.json
```

Clustered-spectrum comparisons and physical shift-invert stopping still need
validation before considering public promotion. The measured candidate and
broader accuracy profile remain explicitly experimental.

## Resolved-cluster comparator correction

A further adversarial test found an error in the initial checkpoint policy
([#109](https://github.com/alecjacobson/warpack/issues/109)). For the diagonal
matrix with eigenvalues `[1000003, 1000002, 1000001, 1000000.001, 1000000, 0]`,
k=4 and tol=1e-10, two supplied complete Ritz sets have exactly zero original
residuals. The second replaces 1000000 with the larger 1000000.001. The old
requested-tolerance tie width (~0.0016) nevertheless discarded that resolved
improvement. The regression failed with a 0.001 target error despite 4/4 counts.
[Failure record](../../results/performance_followup/general-checkpoint-cluster-failure.json).

The comparator now bases ties on the observed maximum residuals and a
32-machine-epsilon floor, scaled by the sum of eigenvalue magnitudes. It no
longer multiplies the requested tolerance by an additional k. This is a
comparison heuristic, not a forward-error bound for arbitrary nonnormal
matrices; fresh original residuals and independent requested-set gates remain
necessary.

All **nine** regressions pass on pinned Warp (5.97 s) and stock 1.15 (6.34 s).
The full 20-case / 60-replay fixed-80 audit was repeated after this correction,
and all cases pass again.
[Corrected-policy results](../../results/general-selection-checkpoint-cluster-fixed.json).
The overhead measurement above belongs to the preceding policy; a final
matched timing control is being recorded separately. Public solver behavior
is still unchanged by this experiment.
