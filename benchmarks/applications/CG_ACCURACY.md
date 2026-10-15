# Inner CG accuracy for shifted FEM modes

This follows the [weighted projection report](MASS_PROJECTION.md) and the
large consistent-mass [FEM comparison](FEM.md), under performance issue
[#15](https://github.com/alecjacobson/warpack/issues/15). Production arithmetic
and default solver settings are unchanged. The FEM harness now exposes
`--cg-tol` explicitly and records it in timing-regression comparisons; changing
it makes a row ineligible for an identical-settings regression claim.

The earlier faster CG profiles passed independent 1e-8 physical accuracy but
often did not report full counts at the outer tolerance 1e-11. Inner tolerance
1e-13 was insufficient for that stricter target. A shifted inverse magnifies
inner solve error differently as the shift and eigenvalue change, and a small
recursive CG residual does not replace the original-pencil residual. The
experiment below tightens the inner tolerance while keeping the same matrices,
upstream preconditioners and tested outer budgets. Every timed trial still faces
original-pencil residual, requested spectrum, M-orthogonality and complete
invariant-subspace gates; strict counts are recorded independently.

All CG uses upstream Warp block-Jacobi or pinned upstream FSAI, with FP64 CG
and FP32 FSAI factors, without a local substitute factorization or CG loop.
FSAI parameters remain row size 32, step size 3 and eight application lanes.
The numerical work is graph captured and stays in Warp. Host reference
construction and validation remain outside timings. Setup, capture, warmup and
three synchronized replay samples are separate in each report. No CuPy or
cuDSS computation is used by these iterative rows.

## Controlled tolerance experiment

On the 11,253-DOF free consistent-mass beam, exclude the six analytic rigid
modes and return six elastic modes. Sigma=-100, ncv=64, one outer cycle:

| Inner CG tolerance | Block-Jacobi median (ms) | FSAI median (ms) | Strict returned counts, all three trials |
|---|---:|---:|---|
| 1e-13 (previous report) | 262.145 | 108.594 | 0/6 for both |
| 1e-14 | 283.928 | 116.723 | **6/6 for both** |
| 1e-15 | 305.229 | 124.522 | **6/6 for both** |

At 1e-14, maximum physical residuals are 4.10e-12 (BJ) and 6.57e-12 (FSAI).
At 1e-15 they fall below 6.6e-13. FSAI's 1e-14 setting adds about 7.5% to
the previous application-tolerance time and recovers strict convergence.
The earlier strictly converged sigma=-1/ncv32 FSAI profile took 199.523 ms;
this is a better tested accuracy/performance configuration, not a change to
CG's algorithm or a new universal default. Warp+cuDSS at the same sigma=-100,
ncv64 budget takes 39.521 ms; the faster strict direct profile with a smaller
basis takes about 17.5 ms. A substantial iterative gap remains.

A separate sigma=-50/ncv48/one-cycle profile with inner tolerance 1e-14 also
passes full counts at 115.112 ms FSAI / 288.864 ms BJ. Its small difference
from 116.723 ms is not evidence of a robust extra speedup. Sigma=-10/ncv32,
inner tolerance 1e-15 passes full counts at 152.703 / 387.414 ms; using fewer
outer vectors alone is not enough to minimize iterative solve cost.

## Scaling and validation

The [composite report](../../results/cg_accuracy_coverage.json) contains **24
configurations, all passing the external numerical gates**. Eighteen reach full
strict counts on all three trials; six initial 1e-14 configurations do not.
Those six were rerun at 1e-15 and all reach full strict counts. They remain
recorded as incomplete strict profiles, not silently overwritten.

Accepted profiles with **full strict counts on every trial**, milliseconds:

| Consistent-mass case | Sigma / ncv / cycles | BJ tolerance / time | FSAI tolerance / time |
|---|---|---|---|
| Clamped, 10,890 DOFs, 6 modes | -100 / 64 / 1 | 1e-15 / 305.084 | 1e-15 / 121.722 |
| Clamped, 41,616 DOFs, 6 modes | -100 / 64 / 1 | 1e-14 / 504.237 | 1e-15 / 288.022 |
| Clamped, 10,890 DOFs, 20 modes | -10 / 64 / 2 | 1e-14 / 1156.388 | 1e-14 / 445.249 |
| Clamped, 10,890 DOFs, 40 modes | -10 / 96 / 2 | 1e-15 / 1706.520 | 1e-15 / 662.653 |
| Free/projected, 11,253 DOFs, 6 modes | -100 / 64 / 1 | 1e-14 / 283.928 | 1e-14 / 116.723 |
| Free/projected, 42,483 DOFs, 6 modes | -100 / 64 / 1 | 1e-14 / 500.255 | 1e-15 / 291.941 |

These are tested profiles, with the same upstream preconditioners and complete
physical-space checks. There is no claim that 1e-14 or 1e-15 universally
ensures an outer tolerance of 1e-11. The original 1e-14 clamped 20-mode FSAI
result approaches that boundary (maximum residual 9.91e-12), making the
per-trial original-problem checks particularly relevant. Every inner solve in
these profiles reports convergence, and pure-Warp graph audits show no HOST
callbacks or host/device transfers.

The added harness argument and regression guard pass **34 focused tests in
33.89 seconds**, including all FEM and weighted-projection tests. Lint, Python
formatting and wheel build pass. The preceding production checkpoint has
256 passing full-suite tests; this follow-up changes no production solver
arithmetic. The unchanged default CG profile is checked separately against its
saved accuracy/graph/timing baseline. Both pass, with replay ratios 1.00008, 0.99983 (BJ, FSAI). See the [raw regression report](../../results/cg_accuracy_default_regression.json) and [validation provenance](../../results/cg_accuracy_validation.json).
This does not rerun or improve Anvil's 283,857-DOF result. Anvil uses different
material scales, a mass-normalized formulation and its own tuned shift/FSAI
profile. The general upstream/complex cases are not rerun by this experiment.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.applications.fem_compare --cases free --side 10 \
  --project-rigid --methods warp-cg-bj,warp-cg-fsai --ncv 64 --steps 1 \
  --shift -100 --cg-tol 1e-14 --out results/cg_accuracy_free.json
```
