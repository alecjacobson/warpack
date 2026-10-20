# Fused projected Jacobi rounds and Anvil block timing

Follow-up to [#83](https://github.com/alecjacobson/warpack/issues/83) and
[#67](https://github.com/alecjacobson/warpack/issues/67). Global Jacobi keeps the
existing pair-angle kernel, then computes both `J.T H J` and `V J` in one
out-of-place Warp kernel. Alternating preallocated buffers removes one launch
per round; two device-to-device copies restore public buffers once per sweep.
The bounded device convergence loop, 30-sweep maximum and 2e-15 off-diagonal
criterion are unchanged. No numerical host work or replay synchronization is
introduced. Existing setup synchronization for driver issue #61 remains.

## Isolated projected solve

L40, three warm batches of ten captured solves, identical restored PSD inputs,
complete dense spectrum/equation and orthogonality checks after every batch.
All samples pass, with uncontended occupancy records. Median milliseconds:

| Columns | Existing shared | Previous global | Fused global |
|---|---:|---:|---:|
| 32 | .844 | 1.779 | 1.381 |
| 40 | 1.831 | 2.466 | 1.909 |
| 48 | 3.075 | 3.217 | 2.473 |
| 60 | 4.805 | 3.757 | 2.895 |
| 64 | 6.068 | 4.671 | 3.576 |
| 72 | 9.889 | 5.322 | 4.098 |
| 96 | — | 7.255 | 5.551 |
| 128 | — | 10.127 | 7.798 |

Ada (architecture 89) therefore uses the bounded global path from 48 columns
when conditional graphs are available. Other architectures retain the former
shared/global dispatch, including the block-specific shared 72-column path.
Odd dimensions and missing-conditional compatibility retain their previous
paths. Fusion applies wherever the bounded global path is already selected.
The thresholds are measured on L40, not claimed universal across GPUs.

Raw [shared](../../results/performance_followup/block-projected-shared.json),
[previous global](../../results/performance_followup/block-projected-global.json),
[large previous global](../../results/performance_followup/fused-projected-large-before.json)
and [fused](../../results/performance_followup/block-projected-fused.json) records
include source hashes, graph inspection and numerical checks.

## Anvil, 283,857 unknowns and 20 elastic modes

The same experimental FSAI32/FP32-CG15 configuration uses Warp width 24,
CuPy width 20, seeded random starts, device-selected CG batches and
cached operator images. Matrix/reference inputs, tolerances and stopping
policies remain unchanged. The reference is loaded from the existing validated
cache; no giant CPU factorization is run. Warm medians of three trials:

| Projected policy | Warp seconds | CuPy seconds |
|---|---:|---:|
| Original shared72, fresh control | 3.309 | 3.166 |
| Original shared72, second control | 3.249 | 3.188 |
| Previous global72, first trial block | 3.243 | 3.104 |
| Previous global72, second trial block | 3.189 | 3.263 |
| Fused global with Ada48+ dispatch | 3.111 | 3.087 |

Every row passes original residual, spectrum, mode-shape and orthogonality
checks. Warp reports all 20 at its normalized tolerance 1e-8; CuPy meets its
fresh full-block absolute tolerance 1e-5. These are different stopping formulas.
The final Warp samples are 3.107/3.111/3.188s and 28/28/29 outer iterations;
CuPy samples are 3.241/3.072/3.087s. Variable reference trajectories preclude a
universal parity claim. This reduces Warp time by 4.2–6.0% against the fresh
original controls; it does not close the iterative-versus-cuDSS gap (#15).

Final maximum original elastic residual is 1.997e-8, returned-pair residual
9.985e-9. Scheduled inner vector iterations are 8,640–8,700; unfinished inner
CG applications are intentional approximate preconditioning. Setup is 1.202s,
capture .592s, and setup/capture device free-memory decrease 4.308GB (not peak
memory). All are separate from replay. The captured graph contains only kernels,
conditionals, memsets and device-to-device copies; no HOST/HtoD/DtoH nodes.

Continued-step instrumented medians reduce the metric eigensolve from 5.121 to
3.791ms and Rayleigh–Ritz eigensolve from 5.304 to 2.212ms. Whole instrumented
step time falls from 70.054 to 65.824ms. These overlapping categories are not
additive, and this continued-step profile is distinct from adaptive solve time.
Post-profile original equations, full counts and status pass too.

[Original control](../../results/performance_followup/anvil-projection-control.json),
[original profiled control](../../results/performance_followup/anvil-projection-control-profile.json),
[global72](../../results/performance_followup/anvil-projection-global.json),
[global72 profiled](../../results/performance_followup/anvil-projection-global-profile.json),
[final fused](../../results/performance_followup/anvil-projection-fused.json).
Associated `.occupancy.json` records retain the timing eligibility checks.

## Benchmark acceptance

[#84](https://github.com/alecjacobson/warpack/issues/84) makes block and related
FEM comparison jobs return nonzero when any declared numerical/stopping gate
fails. Physical/outer and fresh native full-block results remain separate;
all rejected trials are retained. Post-profile status/counts are mandatory.
Intentionally unfinished truncated-CG preconditioners are not mislabeled as
failed inverse solves. All installed core module hashes are now recorded.
Historical raw reports retain their original schema and per-trial evidence.

## Related starts and other solve paths

The 10,890-DOF consistent-mass beam applies a regional 2% stiffness change,
with previously computed modes/full blocks as starts. The same FSAI32/CG10,
width 24, 20 modes and unchanged physical/count gates pass every trial:

| Start | Previous Warp, ms | Fused Warp, ms | Fresh CuPy, ms |
|---|---:|---:|---:|
| Random | 292.40 | 196.67 | 414.88 |
| Previous modes | 221.15 | 150.37 | 321.82 |
| Previous full block | 173.20 | 124.54 | 316.81 |
| Previous full block + explicit exploration | 187.74 | 142.62 | — |

[Fresh per-trial data](../../results/performance_followup/related-consistent-fused.json)
includes every native full-block tolerance result. Starts computed by the
updated Warp solver can change the native iteration trajectory; the fresh
CuPy full-block result is slower than the earlier 220.71ms record. Retain both;
do not attribute that reference change to a CuPy code modification. Explicit
exploration remains a separate policy, with no matched native claim.

Other uncontended, three-trial controls retain full accuracy gates:

- 100000×64 clustered POD: **12.319ms**, previously 14.575ms; native CuPy
  **93.536ms**. [Data](../../results/performance_followup/fused-normal-64.json).
- 100000×128 clustered POD: **73.870ms**, previously 75.391ms; native CuPy
  **160.139ms**. [Data](../../results/performance_followup/fused-normal-128.json).
- 27,648-DOF regular LM with exact diagonal mass inverse: **187.723ms**,
  previously 201.648ms; scaled Warp **177.947ms**, previously 191.984ms;
  native CuPy eigsh **118.092ms**. [Data](../../results/performance_followup/fused-regular-lm.json).
  The diagonal route meets both its declared tolerance and the extra native
  absolute threshold. One scaled trial still exceeds that additional native
  absolute threshold, as in the previous report; it passes its own normalized
  stopping and all common physical gates. Do not claim identical stopping.
- Anvil Warp+cuDSS remains near its existing .467s control, with all 26 original
  mode checks passing. [Data](../../results/performance_followup/fused-anvil-direct.json).
  Its graph retains the known vendor metadata HtoD copies (#29), with no host
  callbacks or device-to-host copies. The main all-backend table is unchanged.

## Validation and retained failures

The full local run passes **839 tests in 400.59s**. A final benchmark-only
acceptance addition passes with all four gate tests (**.69s**); it was added
after collection of that full run. Stock Warp 1.15 passes **158 focused tests
in 29.24s**, including that addition. Coverage includes independent dense
Jacobi rotations, difficult complete projected spectra, changed inputs on
captured graphs, repeated block eigenspaces, coupled mass, invalid data,
compatibility dispatch and no-host-transfer graph audits. Lint and formatting
pass. The clean wheel imports all 23 core modules outside the source checkout.
[Validation record](../../results/performance_followup/fused-projected-validation.json),
[wheel record](../../results/performance_followup/fused-projected-wheel.json).

The deliberately one-step block solve reports **0/4** and exits1 while retaining
its [failed numerical row](../../results/performance_followup/fused-block-rejected-cg.json).
Two earlier attempts instead exposed a setup bug in the public adapter for
upstream FP64 BSR preconditioners ([#86](https://github.com/alecjacobson/warpack/issues/86));
these are separately retained as [block-Jacobi](../../results/performance_followup/fused-block-rejected.json)
and [FSAI](../../results/performance_followup/fused-block-rejected-fsai.json)
setup errors. Neither counts as the successful rejection-gate test. The measured
CG-wrapped preconditioners are unaffected. Broader composite-driver exit-status
coverage is tracked in [#85](https://github.com/alecjacobson/warpack/issues/85).
