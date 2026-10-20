# Captured block finalization and a largest-mode experiment

[#87](https://github.com/alecjacobson/warpack/issues/87) adds the optional
`finalizer` argument to `BlockLOBPCG.capture`, matching the symmetric/Krylov API.
It warms device postprocessing before capture and records it once after either
the fixed or adaptive loop. Default iteration and stopping are unchanged.
Allocate outputs before capture; the callback records graph-capturable device
operations during setup. It is not a host callback executed during replay.

The benchmark-only `LargestBlock` adapter uses the existing
`SparseOperator(K, sign=-1)` to solve `-K x = mu M x`, then restores
`lambda = -mu` in a Warp finalizer. This selects **largest algebraic** modes;
it is not arbitrary largest-magnitude selection. Small indefinite regressions
include large negative eigenvalues so LA and LM cannot be confused. The public
block API continues to target SA; application reformulation stays outside core.

## Replay and numerical validation

**58 focused tests pass in 7.36s** on pinned Warp and **7.81s** on stock Warp 1.15.
They include existing public/experimental block regressions plus:

- Fixed/adaptive postprocessing runs once per replay and observes final
  values, counts and status. Non-finite inputs remain rejected; restoring
  coefficients restores convergence on the same graph.
- Standard/diagonal/coupled metric problems, complete repeated largest spaces,
  changed coefficients and starts, original equations, metric orthogonality,
  source preservation and no HOST/HtoD/DtoH graph nodes.
- Independent dense checks that the model-derived upper bound is valid and the
  separately assembled `sigma M-K` matrix matches its specification, is SPD,
  and leaves inputs unchanged.

A rejected prototype tried to launch an already captured conditional graph
inside another stream capture; CUDA rejected that unsupported composition.
The finalizer records operations directly in the original graph instead.
No extra solver iteration or host sign recovery is needed.
[Validation record](../../results/performance_followup/largest-block-validation.json).
This is focused validation following the preceding full local suite, not a
new full-suite run.

## Large regular-FEM experiment (#31)

The existing [negative-foundation Q1 fixture](../applications/NEGATIVE_FEM.md)
has 27,648 unknowns, 20 modes and repeated polarizations. An independent
analytic check establishes that its LM target coincides with LA; reference
vectors never enter initialization. The fixture is a negative-curvature
surrogate, not geometric prestress or a buckling load-factor pencil.

A first mass-diagonal-preconditioned block profile is not competitive: lumped
mass needs 259–264 iterations and about 2.27s, while consistent mass fails at
300 iterations. [Retained rejected comparison](../../results/performance_followup/largest-block-initial.json).
The successful lumped row is a slower diagnostic, not an improved setting.

A stronger preconditioner uses an upper bound from the **input Q1 stencil**.
With h=1/(side+1), each one-dimensional generalized eigenvalue is below 4/h²
for lumped mass or 12/h² for consistent mass. The three material weights sum
to 5, so `sigma = (20 or 60)/h² - foundation` is an upper bound. Warp constructs
`sigma M-K`; unmodified upstream FSAI32 and eight batched FP32 CG iterations
approximate its inverse. These are preconditioning applications, so unfinished
inner solves are diagnostic rather than exact-inverse acceptance failures.
The bound is specific to this fixture; no universal automatic shift is added.

Accepted three-trial uncontended warm results, width 24, outer tolerance 1e-13:

| Mass | Pure-Warp block median | Outer iterations | Fresh absolute residual gate |
|---|---:|---:|---:|
| Lumped | 346.886ms | 30 | scaled Frobenius <=1e-8 |
| Consistent | 432.180ms | 31–32 | original max column norm <=1e-8 |

Every trial also passes the unchanged original residual/spectrum, complete
requested space and metric orthogonality bounds, full 20 counts and status 0.
Lumped scaled Frobenius residual is at most 2.30e-9; consistent max column
residual is at most 9.88e-12. The graphs contain only device operations and
DtoD copies. Shifted-matrix/factor/solver setup, uploads and compilation where
encountered are separate from capture and warm replay. Setup differences
between sequential rows include cache effects, not intrinsic mass-type cost.
[Accepted profiles](../../results/performance_followup/largest-block-upper-cg8.json).

This loses on lumped mass to the preceding accepted exact-diagonal Krylov
profile 187.723ms and native CuPy eigsh 118.092ms. On consistent mass, a
[separate accepted fresh Krylov+cuDSS control](../../results/performance_followup/largest-consistent-krylov-control.json)
takes 592.233ms: the 432.180ms block profile is about 27% faster for this workload,
with a stronger, fixture-specific preconditioner and a tighter outer tolerance.
It is not a general replacement for Krylov or evidence of a universal default.

The [fresh multi-method comparison](../../results/performance_followup/largest-block-upper-cg8-references.json)
rechecks lumped Krylov/CuPy at 187.805/120.374ms and consistent block/Krylov at
420.607/593.041ms, with those individual rows passing. The **overall comparison
is rejected** because CuPy's consistent-mass LOBPCG+cuDSS reference exhausts its
200-iteration profile with physical residual up to 4.90e-7, requested-space
error 4.98e-5 and a missed native absolute bound. That time is not an accepted
speed baseline. This establishes nonconvergence at that budget, not by itself
proof of the long-run instability tracked in #80. The independent analytic
reference and both Warp routes agree; the failed native rows remain unchanged.

Reproduce the stronger profile without optional reference methods:

```bash
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m benchmarks.experimental.largest_block_compare --side 24 \
  --modes 20 --width 24 --steps 300 --tol 1e-13 --preconditioner upper-cg \
  --inner-cg 8 --repeats 3 --out build/largest-block.json
```

Exact executed commands and source hashes are retained in the raw
occupancy/validation files.
#31 remains open for the remaining regular/adaptive throughput gap. #87 closes
only the captured-finalization API gap.
