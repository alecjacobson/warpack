# Cooperative projected Cholesky qualification (#154)

The 100-mode FEM profile identifies a persistent cost in factoring the full
104-column basis metric, even after the active residual block becomes small.
See [the active-step profile](../fem_100_modes/PROFILE.md). This study changes
only that small Cholesky factorization; triangular application, projected
eigensolves, Gram products, stopping tests and preconditioning stay fixed.

The candidate computes independent entries of each column cooperatively in
Warp, storing the packed lower triangle in shared memory. Each entry retains
the serial kernel's subtraction order, invalid/nonpositive pivot flag and
diagonal containment. It uses no vendor dense solver or host numerical work.
This commit qualifies the benchmark prototype; production dispatch is unchanged.

## Isolated factorization

L40, FP64, pinned Warp 1.18 overlay; three warm captured batches of 100 factors
at each dimension. These are factorization times, excluding setup and capture.
Every measured batch passes independent matrix reconstruction, positive
diagonal and status checks. Captured graphs contain no host callback or host
transfer. The external occupancy monitor found no competing GPU process.

| Dimension | Serial (µs) | Cooperative (µs) |
| ---: | ---: | ---: |
| 4 | 5.837 | 6.872 |
| 8 | 16.241 | 11.453 |
| 16 | 64.755 | 21.679 |
| 24 | 162.737 | 34.824 |
| 48 | 878.252 | 79.080 |
| 64 | 1840.710 | 115.543 |
| 104 | 6709.743 | 234.092 |
| 128 | 11857.613 | 325.523 |

Raw evidence: [factors](isolated/factor-a1.json),
[occupancy](isolated/factor-a1.occupancy.json). The dimension-4 regression calls
for a small-dimension fallback if the candidate is promoted. Shared-memory
capacity and non-CUDA devices also require a fallback. The 28.7× factor speedup
at dimension 104 is not a whole-solve speedup.

The direct regression tests exercise scaled SPD matrices, strided buffers,
changed-input graph replay, invalid inputs and recovery. There are 18 passing
direct tests on both stock Warp 1.15 and the pinned overlay. Three additional
coupled/free/heterogeneous FEM integration tests pass on the overlay; stock
Warp skips these because it lacks the upstream FSAI preconditioner. The active
profiler preservation test also passes.

## Whole-solve qualification

Matched serial/cooperative 100-mode FEM and 20-mode Anvil comparisons are in
`paired/`. Both variants use the same fixed setup and seed, existing
independent spectrum/subspace checks and native convergence counts, with three
measured solves. The CuPy comparator is run for both. Anvil uses the existing
validated reference cache. No large CPU factorization is rerun.

| Problem | Factor | Warp construction + capture (s) | Warp warm median (s) | CuPy warm median (s) |
| --- | --- | ---: | ---: | ---: |
| 34,560 DOFs, 100 modes, width 104 | Serial | 4.6798 | 3.7079 | 3.5637 |
| Same wide FEM | Cooperative | 4.3561 | 3.2463 | 3.8488 |
| Anvil, 20 elastic modes, width 24 | Serial | 1.6217 | 1.6066 | 2.3823 |
| Same Anvil | Cooperative | 1.6750 | 1.6274 | 2.3875 |

These are fresh within-study controls, not a replacement for the README's
six-backend showcase. The paired source, dependency, matrix/reference/starting
vector identities agree exactly. Setup is one construction plus capture per
worker, excludes warmup, and is not a repeated setup median. It includes the
preconditioner and solver workspaces. CuPy reuses that worker's preconditioner
for a matched replay comparison, so this table makes no independent CuPy setup
claim. Its separately recorded adapter/upload costs are 0.0840/0.0844 seconds
for wide FEM and 0.2569/0.2544 seconds for Anvil (serial/cooperative workers).

The wide Warp median improves by **12.4%**. All requested modes pass the unchanged
native and independent physical checks: worst Warp residual 7.81e-12, requested
subspace error 1.38e-11. Warp takes 34–35 outer steps. CuPy takes 159–260 history
steps and varies from 2.734 to 4.016 seconds across the two workers; its median
comparison is not evidence of a universal Warp advantage. This variability
and remaining projected eigensolve cost leave #147 open.

Anvil has no demonstrated improvement: the cooperative median is 1.3% slower,
with individual Warp runs spanning 1.603–1.672 seconds. All six Warp and six
CuPy solves pass; worst original residuals are 1.99e-8 and 1.81e-9 respectively,
including six reconstructed rigid modes. **Keep the serial path for the Anvil
width.** Promotion should target wide blocks where the factor cost matters.

Qualified workers: `paired/wide-serial-a3`, `paired/wide-cooperative-a1`,
`paired/anvil-serial-a2`, `paired/anvil-cooperative-a2` (JSON plus occupancy).
`wide-serial-a1/a2` are externally contended and excluded even if the worker's
before/after spot checks miss the overlap. The first Anvil attempts failed
before solving because the external fixture path was not set; the retry uses
`WARPACK_ANVIL_INPUTS=/run/openclaw/horde/warpack-anvil-inputs-142.npz`.
The worker records warmup duration but does not independently validate warmup.

`wide-screen.json` is a single-repeat accuracy screen (100 accepted modes in
34 outer iterations), not the repeated performance comparison. Its original
residual is 6.80e-12 and reference subspace error is 1.63e-11.
Contended attempts are retained and excluded from timing conclusions.
