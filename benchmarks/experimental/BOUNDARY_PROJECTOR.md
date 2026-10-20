# Implicit boundary products for conformal modes

The [exact conformal pencil](../applications/CONFORMAL.md) has a centered
boundary mass `M=diag(C,C)`, where `C=D-b b.T/nb`. Explicit CSR storage gives
long, dense boundary rows. The experimental product represents each block with
one boundary sum and a pointwise update instead. The shifted product uses
sparse `K-sigma D` plus the rank-two term `sigma b b.T/nb`.

Both representations apply the same unregularized pencil and translation
gauge. The original explicit shifted matrix still constructs the upstream
FSAI factor or cuDSS factor, and the finalizer still checks full original CSR
products. This change accelerates replay products; it does **not** eliminate
the explicit setup matrices or claim a factorization/storage improvement.
All added numerical kernels are Warp. No installed core source changes.

## Matched large comparison

NVIDIA L40, the same pinned Warp 1.18/FSAI environment as the conformal report;
32,766 reduced unknowns, 20 modes, ncv64, four fixed cycles, shift `-0.001`.
CG uses FP64 arithmetic, the unchanged upstream FP32 FSAI32 factor, tolerance
`1e-9`, zero absolute tolerance and one relative-0.01 residual correction.
Three synchronized trials per row, sequential isolated workers, no sampled
competing GPU process. All original accuracy and strict stopping gates pass.

| Product / inverse | Setup (s) | Capture (s) | Solve median (s) |
|---|---:|---:|---:|
| Explicit / cuDSS | 0.4830 | 0.6399 | 0.418504 |
| Implicit / cuDSS | 0.4925 | 0.7455 | 0.383698 |
| Explicit / corrected CG | 1.4927 | 5.9511 | 12.686540 |
| Implicit / corrected CG | 1.5210 | 4.3992 | 6.801015 |

Implicit products reduce the measured CG solve by **46.4% (1.87x)** and the
cuDSS solve by **8.3% (1.09x)**. CG work is almost unchanged: 136,297 versus
135,836 accumulated iterations over 187 RHSs, a 0.34% difference. Floating-point
product order changes some iteration histories; this is a whole-solve
comparison, not a kernel-only attribution. Maximum fresh relative linear
residuals are `1.05e-11` explicit and `9.70e-12` implicit; neither has a failed
audit. Both recover all 20 physical and reciprocal counts at `1e-10`.

The default explicit control graph summaries are identical to the preceding
coverage records. Pure-Warp graphs contain no HOST/HtoD/DtoH nodes. cuDSS
retains vendor metadata transfers (#29). The remaining iterative gap stays
open in [#98](https://github.com/alecjacobson/warpack/issues/98).

## Runtime fingerprint repair

[#100](https://github.com/alecjacobson/warpack/issues/100) found that hashing
`warp.optim.linear` alone fingerprints a forwarding module in the pinned Warp
checkout. The shared helper now also hashes the actual exported CG,
preconditioner, FSAI and sparse-matvec implementation files. Stock Warp's
unavailable FSAI is recorded explicitly. Connection and conformal baseline
gates reject legacy reports that lack these implementation hashes, even when
two legacy dictionaries happen to match. Old records are preserved unchanged;
this finding is not evidence that the old runs mixed runtimes.

The baseline and regression for this report carry the strengthened fingerprints.
The earlier implicit screens remain useful numerical/timing diagnostics but
are not automatically comparable under the new fingerprint policy.

## Tests and reproduction

Fourteen new operator checks cover boundary/interior gauges, multi-chunk
boundary reductions, scaled/changed RHSs, multi-row and in-place products,
LinearOperator alpha/beta with separate buffers or output aliased to x, y, or both, captured replay,
and flat/curved modes with CG/cuDSS. Runtime tests detect an implementation-file
change behind an unchanged forwarding module and reject incomplete legacy
fingerprints. **69 focused checks pass on pinned Warp 1.18 and stock Warp 1.15**.
The production core and installed package are unchanged.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.monitor_gpu \
  --out build/products.occupancy.json --log build/products.log -- \
  python -m benchmarks.applications.conformal_compare \
  --side 128 --modes 20 --ncv 64 --steps 4 --shift -0.001 --cg-tol 1e-9 \
  --repeats 3 \
  --methods warp-cudss,warp-cg-fsai,warp-cudss-implicit,warp-cg-fsai-implicit \
  --out build/products.json
```

For regression, use only the two implicit methods and
`--baseline results/conformal-products.json`. The original explicit methods
and defaults remain available. The optional constructor flag is
`ConformalEigensolver(..., implicit=True)` in the benchmark adapter.

- [Matched results](../../results/conformal-products.json), [occupancy](../../results/conformal-products.occupancy.json).
- [Regression](../../results/conformal-products-regression.json), [occupancy](../../results/conformal-products-regression.occupancy.json).
- [Initial screen](../../results/conformal-implicit-screen.json) and [pre-fingerprint repeated profile](../../results/conformal-implicit.json).

The strengthened same-configuration regression passes at **0.383731 s** for
cuDSS (ratio **1.000086**) and **6.801272 s** for CG (ratio **1.000038**).
All three trials per method pass every original/strict gate; the monitored GPU
is uncontended. [Validation record](../../results/performance_followup/conformal-products-validation.json).
