# Strict generalized-pencil accuracy diagnostics (#123)

**No recovery is promoted.** All twelve large-worker configurations reject the
unchanged strict native tolerance on at least one output. The small canary passes.
The 52 first/later diagnostic outputs and every rejected result remain archived.
These are accuracy controls, not timing comparisons.

The procedural consistent-mass chain has 65,536 first-order unknowns, 20 requested
finite modes, ncv 64, two fixed Arnoldi cycles, shift -0.15+6.3i and original-pencil
tolerance 1e-9. Both nonproportional damping and gyroscopic variants use the same
existing public solver and optional cuDSS inverse. Corrections inside each sparse
solve use existing Warp residual products/updates; cuDSS only applies its factors.
No numerical core/default changes are introduced by these controls.

## Sparse-solve corrections

Only the declared fixed correction budget changes. Original eigenvalues and
eigenvectors receive no postprocessing. Every first solve and three later graph
replays check original equations, requested spectrum/subspace, independence and
full native count/status.

| Kind | Corrections per inverse | Native counts, first then later | Worst native residual |
|---|---:|---|---:|
|[nonproportional](16384-nonproportional-steps1.json)|1|18 / 18 / 20 / 17|1.929e-09|
|[nonproportional](16384-nonproportional-steps2.json)|2|18 / 20 / 18 / 18|1.588e-09|
|[nonproportional](16384-nonproportional-steps4.json)|4|19 / 19 / 19 / 20|1.170e-09|
|[gyroscopic](16384-gyroscopic-steps1.json)|1|18 / 19 / 18 / 18|1.881e-09|
|[gyroscopic](16384-gyroscopic-steps2.json)|2|20 / 19 / 18 / 19|1.933e-09|
|[gyroscopic](16384-gyroscopic-steps4.json)|4|19 / 20 / 20 / 20|1.257e-09|

Extra correction occasionally yields 20 passing pairs, but not repeatably across
all four outputs. The residual is dominated by the second (dynamic) block; the
first block contributes less than 1e-11 on this normalized scale. Wider-precision
evaluation of the **same stored A/B and returned vectors** agrees with the native
residual. The strict miss is actual returned-vector error, not merely FP64
residual-evaluation noise. This does not uniquely locate every error source.

## Repeated independent inverse filtering

A second explicit control uses one sparse-solve correction and applies
`(A-sigma B)^-1 B` to each final vector 1, 2 or 4 times, normalizing after every
application with the existing Warp kernels. It preserves the original eigenvalues;
no quotient update or mechanics-specific velocity reconstruction is used.

| Kind | Vector polish steps | Native counts, first then later | Worst native residual |
|---|---:|---|---:|
|[nonproportional](polish/16384-nonproportional-steps1.json)|1|19 / 19 / 18 / 18|2.517e-09|
|[nonproportional](polish/16384-nonproportional-steps2.json)|2|14 / 14 / 14 / 14|5.628e-08|
|[nonproportional](polish/16384-nonproportional-steps4.json)|4|3 / 4 / 3 / 4|4.107e-05|
|[gyroscopic](polish/16384-gyroscopic-steps1.json)|1|18 / 18 / 18 / 18|3.675e-09|
|[gyroscopic](polish/16384-gyroscopic-steps2.json)|2|14 / 14 / 14 / 14|5.230e-08|
|[gyroscopic](polish/16384-gyroscopic-steps4.json)|4|4 / 4 / 3 / 4|4.223e-05|

One application reduces the dynamic-block error to about 1e-12 but leaves the
first-block relation inaccurate. Repetition worsens the complete original
residual, reaching roughly 4e-5 after four steps. Amplification of unwanted
components by independent inverse filtering is a plausible interpretation; the
experiment does not uniquely establish that mechanism. More independent polish
steps are not a generic fix. The previous application-specific reconstruction
remains separate from the public solver.

## Independent reference check

The original analytic modal reference remains the acceptance gate. A separate
CPU reference reduces the **actual stored** constant 2x2 diagonal/off-diagonal
blocks. It first verifies exact chain structure and the first-order linearization,
then evaluates `D + 2 cos(theta) E` as `(D+2E) - 4 sin(theta/2)^2 E` using 63-bit
long-double mantissas on this machine, before small dense modal solves. This
accounts for rounding of small foundation terms during physical assembly.

The resulting requested eigenvalues differ from the original analytic reference
by at most 3.21e-10 relatively. That effect is recorded separately and does not
relax or replace any gate. Original-pencil residuals do not depend on a reference
eigensolver, and their strict failures remain under wider-precision evaluation.

Eight CPU tests pass: assembled-modal values and left/right equations against
independent dense pencils, exact structure/linearization rejection, and residual
block splitting. Six additional driver tests verify explicit polish selection,
state restoration and preservation of rejected reports.
[Reference tests](validation/reference-tests.log),
[driver tests](validation/polish-driver-tests.log). The small 128-unknown/eight-mode
[canary](32-nonproportional-steps2.json) passes all four outputs, with residual below
3.7e-14. No foreign GPU process appears in the occupancy samples. Large-worker
nonzero exits and `timing_eligible=false` reflect numerical rejection; no failed
row is used as a speed comparison.

Reproduce with `benchmarks.experimental.pencil_refinement` and explicit
`--refinement-steps 1`, `2` or `4`. The second control uses
`benchmarks.experimental.pencil_polish_control --refinement-steps 1 --polish-steps N`.
Each adjacent occupancy JSON retains the full command, and each report records
exact inputs, source/dependency identities, original/native and extended checks.
#123 remains open for a generic correction or justified attainable-accuracy route.
