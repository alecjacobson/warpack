# Complete coordinate-basis projection for small POD spaces

Follow-up to [#31](https://github.com/alecjacobson/warpack/issues/31). When the
normal search space spans the entire smaller matrix dimension, a complete
coordinate basis can replace serial Krylov expansion. The prototype applies
the normal operator to all coordinate vectors in one batch, then uses the
existing Warp projected eigensolve and guarded SVD finalization. Identity is
an input-independent complete basis; reference singular vectors never enter
initialization. Partial search spaces retain the independent-chain guard.

This explicitly forms the small normal matrix through projection. Normal
products still square conditioning. Original-equation, orthogonality and
unresolved-value checks remain unchanged, including captured recovery after
changing an invalid input to a valid one. There are no host numerical decisions
or replay host transfers. The prototype accepts only a full smaller basis.

## Accepted comparisons

Sequential L40 runs, three warm trials, four CPU BLAS threads. Every row passes
the existing spectrum, complete requested subspaces, both SVD equations,
orthogonality and reconstruction-energy gates. Warp also reports 20/20 paired
counts and status 0. Setup/capture/host reference/validation are excluded and
recorded separately; occupancy samples are clean.

| Clustered snapshot matrix, k=20 | Normal Krylov, ms | Full coordinate basis, ms |
|---|---:|---:|
| 100,000×48 | 15.758 | 8.571 |
| 100,000×64 | 27.213 | 14.548 |
| 100,000×128 | 75.395 | 76.857 |

The 48/128 controls are the current [projected-solver follow-up](../upstream/PROJECTED_CONVERGENCE.md),
and the 64-column control is a fresh all-method comparison. All three use the
same corresponding full normal-space widths. At 128 columns there is no gain;
the larger prototype remains a diagnostic, not a universal replacement policy.

The 64-column comparison uses native basis width 48 and full normal width 64:

| Method | Median ms |
|---|---:|
| Full coordinate normal Warp | 14.548 |
| Krylov normal Warp | 27.213 |
| Native CuPy svds | 93.487 |
| Augmented Warp | 259.510 |
| SciPy ARPACK svds | 738.147 |
| Spectra | 900.052 |

[64-column all-method control](../../results/performance_followup/full-normal-64-control.json),
[48-column coordinate projection](../../results/performance_followup/full-normal-corrected-48.json),
[64-column coordinate projection](../../results/performance_followup/full-normal-corrected-64.json),
[128-column coordinate projection](../../results/performance_followup/full-normal-corrected-128.json).
Core hashes and matrix hashes are recorded. These measurements include the
projected Jacobi accuracy repair #77. Earlier accepted largest-only prototype
controls at 8.562/74.135 ms are retained in `full-normal-products-48/128.json`;
they predate that repair and do not validate its complete projected spectrum.

Ten focused regressions pass on pinned Warp (**2.21 s**) and stock Warp 1.15
(**2.65 s**). They cover tall/wide matrices, complete repeated requested spaces,
changed/restored coefficients, original paired equations, exact caller-data
preservation, rank-floor rejection/recovery and graph transfer audits. Tests
use independent analytic/dense references only for validation.

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.normal_pod --case clustered --size 100000 \
  --modes 20 --snapshots 64 --ncv 64 --reference-ncv 48 --steps 100 \
  --repeats 3 --full-basis-products --compare '' --out build/full-normal64.json
```

The explicit experimental flag permits wider complete projections for further
investigation. The following implementation change limits automatic use to
complete spaces of at most 64 dimensions; larger and partial searches retain
the existing Krylov path. It does not change the augmented `PartialSVD` default
or solve the remaining general adaptive-throughput issue.

## Installed API follow-up

`NormalSVD` now selects the coordinate path when the chosen search space is
complete and has at most 64 dimensions. Larger and partial spaces retain Krylov
expansion; the experimental adapter can force a larger complete projection.

Post-promotion controls (three clean warm trials each):

| Fixture, k=20 | Median ms | Path |
|---|---:|---|
| [clustered, 100000×48](../../results/performance_followup/small-normal-api-clustered-48.json) | 8.567 | complete_coordinate_projection |
| [clustered, 100000×64](../../results/performance_followup/small-normal-api-clustered-64.json) | 14.575 | complete_coordinate_projection |
| [clustered, 100000×128](../../results/performance_followup/small-normal-api-clustered-128.json) | 75.391 | matrix_free_krylov |
| [wide, 48×100000](../../results/performance_followup/small-normal-api-wide-48.json) | 8.628 | complete_coordinate_projection |
| [wave, 100000×48](../../results/performance_followup/small-normal-api-wave-48.json) | 9.324 | complete_coordinate_projection |

All trials pass the independent original-equation, spectrum, requested-space,
orthogonality and reconstruction gates, with status 0 and 20/20 paired counts.
The small complete path records 54 kernels, one conditional and one device-to-device
copy; its graph audit contains no HOST nodes or host/device transfers. The
128-column control retains the Krylov path and 75.39 ms replay. These are warm
solve timings on unlocked clocks; setup and capture are recorded separately.

The expanded focused validation passes **59 tests in 5.64 s** on each tested
Warp version (pinned 1.18 and stock 1.15): 18 complete-basis cases, 35 normal SVD
cases and six SVD replay cases. Small cases exercise the public automatic path;
96-dimensional cases exercise the explicit larger experimental projection.

The [combined local suite](../../results/performance_followup/small-normal-validation.json)
passes **750 tests in 338.29 s**, with lint and formatting checks passing. An
[isolated wheel check](../../results/performance_followup/small-normal-wheel.json)
imports the installed API with 23 core modules and no benchmark dependencies.
