# Skinning: recovering multipliers from cached inverse products

The large skinning comparison now reaches practical parity with adapted
CuPy+cuDSS. Warpack retains its device equilibrium and constraint checks.
The improvement removes redundant inverse work; no tolerances, starting vectors,
Krylov dimensions, restart budgets, or acceptance checks were relaxed.
Tracking: [#56](https://github.com/alecjacobson/warpack/issues/56).

## Diagnosis on shared factors

The [profile harness](skinning_profile.py) constructs one cuDSS factorization
and supplies it to both solvers, with the same physical starting direction.
The input has 45,625 vertices, 14 independent constraints, 20 requested modes,
ncv=64 and sigma=-0.1. Warpack uses two fixed restart cycles. CuPy uses adaptive
eigsh on the mass-scaled KKT inverse; this backend is supplied by our adapter.

On the L40 with Warp 1.18.0/FSAI overlay and CuPy 14.2.0, warm median wall times
exclude setup, capture and independent host validation. CUDA event profiling
runs separately and adds instrumentation overhead. Nested event totals overlap
and must not be added; this is not a CUDA API synchronization trace.

| Measurement | Before | Cached multiplier images |
|---|---:|---:|
| Warpack uninstrumented replay | 338.167 ms | 287.672 ms |
| CuPy uninstrumented solve | 287.876 ms | 289.130 ms |
| Warpack inverse applications | 125 | 105 |
| CuPy inverse applications | 108 | 108 |
| Warpack finalization, instrumented | 51.172 ms | 0.396 ms |
| Warpack inverse work, instrumented | 316.313 ms | 265.886 ms |

Each inverse application includes one cuDSS solve and one residual-correction
solve. The previous finalization performed 20 extra inverse applications solely
to recover multipliers. Removing those applications accounts for essentially
the entire large-case improvement. Remaining instrumented Warp costs include
about 12.1 ms for ordinary mass orthogonalization and 7.0 ms for projected
eigensolves; inverse applications still dominate.

Sources: [before](../../results/skinning_profile_before.json),
[after](../../results/skinning_profile_after.json). Each records numerical source
hash, all trial checks, invocation counts and device/runtime details.

## Implementation and accuracy

Every expansion already computes `[z; nu] = KKT^-1 [M q; 0]`. The solver now
retains the small `nu` block for each basis vector, applies the same Ritz
rotation used for physical vectors, and matches the restart-seed permutation.
For a final vector w, linearity gives its multiplier image without another
inverse application, and `mu = (lambda-sigma) nu` recovers its multiplier.

The cache has `(ncv+keep)*r` FP64 entries, about 9.6 KiB on this case. It is
rebuilt by each initialization and carried across restarts within that solve.
Finalization still explicitly evaluates H w, M w, C w and C.T mu, and both
physical residuals still gate convergence. This requires a fixed linear inverse
during the solve, like the existing physical product cache. No constraint Gram
matrix or normal-equation solve is introduced, and C need not have orthonormal rows.

New regressions compare cached images and final multipliers against independent
dense KKT solves after every restart, including differently scaled/nonorthogonal
constraints, consistent mass and repeated modes. They also assert that
finalization makes no additional inverse calls. Existing captured changed-start,
interior, breakdown, inadmissible-output and skinning-assembly regressions remain.

An [independent repeat using the standard comparison harness](../../results/skinning_reuse_confirmed.json)
measures **288.089 ms Warp / 288.309 ms CuPy**. Both pass every trial; Warp returns
20/20 at its unchanged internal 1e-9 tolerance, with worst independent equilibrium
residual **5.08e-13**. The former published Warp result was 339.187 ms: about
15.1% less replay time. The sub-percent difference from CuPy is practical parity,
not evidence of a robust speed lead. Setup was 0.674 s Warp / 0.837 s CuPy in that
repeat; those single samples include adapter setup and are not an ordering claim.

The standard comparison still times device multiplier/residual evaluation for
Warp and leaves CuPy's independent physical validation outside timing. That
asymmetry is preserved explicitly. cuDSS metadata copies fall from 500 to 420
HtoD graph nodes; zero host callbacks remain. The vendor-copy limitation in
[#29](https://github.com/alecjacobson/warpack/issues/29) is not eliminated.

## Smaller workloads and limits

Unchanged settings against the previous main implementation, milliseconds:

| Vertices / modes / ncv / cycles | Warp before | Warp after | CuPy after |
|---|---:|---:|---:|
| 160 / 6 / 32 / 8 | 35.917 | 35.398 | 23.373 |
| 2,025 / 20 / 64 / 2 | 60.072 | 53.307 | 49.176 |
| 45,625 / 20 / 64 / 2 | 338.167 | 287.672 | 289.130 |

All trials pass numerical gates and full Warp counts. The
[matched regression checks](../../results/skinning_reuse_regression.json)
retain matrix/settings/runtime comparability and the existing 1.25x slowdown
threshold. Small cases remain slower than CuPy; their cycle budgets were not
retuned. These results do not establish improvements to unconstrained problems,
Anvil, complex solvers or the pure-Warp constrained iterative inverse.
No competing GPU process was observed at startup; this is not continuous
contention monitoring. One medium CuPy event-instrumented run was substantially
slower than its uninstrumented trials, so cross-run event totals are diagnostic
only and the table uses uninstrumented wall measurements.

## Reproduction

Use the [pinned environment](README.md). All GPU jobs run sequentially.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.applications.skinning_profile --out results/skinning_profile_after.json
python -m benchmarks.applications.skinning_compare --side 24 --steps 2 \
  --methods cupy-cudss,warp-cudss --baseline results/skinning_large_confirmed.json \
  --out results/skinning_reuse_confirmed.json
python -m pytest -q
```

For the before control, use `warpack/` from commit `b319e4a` with the same
profile harness and environment. Small/medium controls used that exact archived
core in an isolated import directory. Add `--side 3 --modes 6 --ncv 32 --steps 8`
or `--side 8` for those profiles. Reference meshes/matrices and cached independent
solutions are identical. Detailed test/build results are in the
[validation record](../../results/skinning_reuse_validation.json).
