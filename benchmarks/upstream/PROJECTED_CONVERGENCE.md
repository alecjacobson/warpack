# Projected Jacobi convergence budget

Issue [#77](https://github.com/alecjacobson/warpack/issues/77) identified a
common Rayleigh–Ritz accuracy limit: ten Jacobi sweeps need not diagonalize an
FP64 projected matrix to the requested accuracy. This affects the shared-memory
path (even dimensions through 64, plus the block solver's special 72-vector
path) and the global-memory path for larger even dimensions. The existing
odd-dimension serial path is unchanged.

## Failing evidence

The independent fixture uses a seeded orthogonal basis, a repeated leading
value 400, a geometric spectrum down to 0.0025, and two zero values. Every
captured solve restores the same input. The complete spectrum, every original
equation and orthogonality are checked outside timing. Old maximum residuals,
normalized by 400, are:

| Dimension | Residual | Eigenvalue error | Outcome at 1e-11 |
|---|---:|---:|---|
| 48 | 2.731e-10 | 1.734e-14 | fail |
| 64 | 1.671e-7 | 1.439e-11 | fail |
| 72 | 1.563e-7 | 1.527e-9 | fail |
| 96 | 4.008e-7 | 8.287e-9 | fail |
| 128 | 6.809e-6 | 1.022e-7 | fail |

Orthogonality remains below 4e-14. Good orthogonality is insufficient here.
[Failing control](../../results/performance_followup/projected-before.json).
Its timings are diagnostics, not accepted performance comparisons. The original
public eigenpair checks can reject inaccurate requested pairs; this report does
not claim false public convergence or invalidate earlier accepted largest-only
POD comparisons.

## Repair and device control

Both affected paths allow at most **30 sweeps**. The shared path retains its
existing device stop when the maximum off-diagonal entry is at most 2e-15 times
the maximum absolute diagonal entry. The global path now applies the same
criterion through a bounded device conditional loop. Each completed sweep
updates preallocated counters and loop state; nonfinite data cannot create an
unbounded loop. No numerical convergence values are copied to the host.

Capturing an outer solver embeds the loop using preallocated arrays. Eager
calls reuse a separate cached graph. Its first preparation loads modules and
synchronizes once before graph upload, preserving the setup containment for
[#61](https://github.com/alecjacobson/warpack/issues/61). Later eager calls and
captured replays add no host synchronization. Setup/capture remain distinct
from replay timings. CPU and runtimes without conditional graph support use
the fixed 30-sweep global fallback; no existing eager/fixed-graph capability is
removed. The capability check uses runtime metadata, not matrix data.

A bounded sweep budget is not an accuracy guarantee for every possible input.
Original-equation counts and residuals remain necessary. A full search basis
alone does not excuse an inaccurate projected eigensolve.

## Matched-accuracy projected controls

The fixed 30-sweep control passes the unchanged 1e-11 gates for all dimensions.
The adaptive global path also passes and needs **13 sweeps** for each larger
fixture. Sequential L40 medians, three samples averaging ten captured solves:

| Dimension | Fixed 30, ms | Device stop, ms |
|---|---:|---:|
| 48, shared early stop in both | 3.0758 | 3.0760 |
| 64, shared early stop in both | 6.0698 | 6.0679 |
| 72 | 10.8437 | 5.3379 |
| 96 | 14.7444 | 7.3520 |
| 128 | 20.4296 | 10.1074 |

These isolate projected diagonalization, not complete eigenproblem speedups.
All accepted controls have clean occupancy records, full-spectrum validation
on every sample, and graph audits with no HOST, HtoD or DtoH nodes. The global
conditional body is recorded once instead of unrolling 30 copies.
[Fixed accurate control](../../results/performance_followup/projected-fixed30.json),
[device stop](../../results/performance_followup/projected-adaptive.json).
Sibling `.occupancy.json` files contain the monitor samples.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.projected_profile --sizes 48,64,72,96,128 \
  --repeats 3 --batch 10 --out build/projected.json
```

The old control uses numerical source `f00326b`. To reproduce the accurate fixed
control, change the two ten-sweep bounds in `warpack/projected.py` and the global
branch of `warpack/solver.py` to 30, without adding the conditional helper. Raw
reports retain source hashes. The final driver additionally records the actual
global sweep count; validation equations and thresholds are unchanged.

## Regression scope

Twenty new cases cover dimensions 48/64/72/96/128, PSD and indefinite spectra,
scales 1 and 1e-6, repeated leading spaces, zero spaces, changed matrix values
and restoration on the same captured graph. They check every original equation,
complete eigenvalue sets, conjugate-free real orthogonality and both known
repeated subspaces, preserve caller data and audit graph transfers. All 20 pass
in 2.50 s on pinned Warp. Stock Warp 1.15 passes 81 focused tests in 11.80 s,
including these cases, public normal-product SVD/replay and cached block-image
regressions. Further combined and application controls are recorded below.

The combined pinned suite passes **730 tests in 357.34 s**. A subsequent metadata
capability check adds two fallback cases, covering eager and fixed capture when
conditional support is disabled: all **22 projected tests pass in 2.28 s** on
pinned Warp and **2.36 s** on stock Warp 1.15. This simulates the capability
branch on the same L40, not a hardware test on an older driver. The numerical
path used by the full run is unchanged. [Validation/source record](../../results/performance_followup/projected-validation.json).
A [clean-source wheel](../../results/performance_followup/projected-wheel.json)
imports all 23 core modules away from the checkout, without benchmark modules.

## Application regression controls

All runs are sequential, have clean occupancy records, and reuse cached Anvil
reference modes. No large CPU sparse factorization is performed. Three warm
trials per row; setup, capture and validation are recorded separately.

| Anvil configuration | Before, s | Repaired, s |
|---|---:|---:|
| Warp block + FSAI32 / active CG15 / cached images | 3.33123 | 3.29781 |
| Accompanying CuPy LOBPCG + same CG preconditioner | 3.28047 | 3.22764 |
| Warp + cuDSS, shift 1, ncv 64, one cycle | .467716 | .467354 |

These small variations do not establish a speedup or parity with CuPy. Every
block trial passes all 20 requested modes, original equations, eigenvalues,
mode shapes, orthogonality, status and convergence counts. CuPy's fresh returned
vectors meet its native absolute tolerance on all trials. The repaired Warp
block's worst original residual is 1.648e-8 and worst reported-pair residual
8.239e-9. Its graph retains 523 kernels, 13 conditional nodes and only device
copies. [Block before](../../results/performance_followup/projected-anvil-old.json),
[block repaired](../../results/performance_followup/projected-anvil-fixed.json).

The cuDSS controls check all original residual, shape and orthogonality gates
on each trial; the repaired worst original residual is 2.695e-9. This driver
does not record strict inverse-space counts, so it is not a resolution of #36.
Its graph retains 128 vendor HtoD metadata copies and no HOST/DtoH nodes, as
tracked in #29. [Direct before](../../results/performance_followup/projected-cudss-old.json),
[direct repaired](../../results/performance_followup/projected-cudss-fixed.json).
Use `benchmarks.cg_experiments --mode eigen --shifts 1 --preconditioners cudss
--ncv 64 --repeats 3 --profile` to reproduce this GPU-only control with cached
`build/anvil_reference.npz` and `build/anvil_initial.npy`. It does not invoke
the legacy all-backend Anvil benchmark.

Public normal-product POD controls pass all original gates and 20/20 counts:
100,000×48 changes **15.7528→15.7577 ms** and 100,000×128 changes
**74.9519→75.3945 ms**, both under 1%. The 128-vector graph shrinks from
6,410 to 2,985 kernel nodes, adds one conditional node, and retains only device
copies. Its process-local capture measurement changes .5311→.3537 s; these
single setup samples are not a first-use speedup claim.
[48-column control](../../results/performance_followup/projected-pod-48.json),
[128-column control](../../results/performance_followup/projected-pod-128.json).
The original matrix hashes and requested subspaces are unchanged.
