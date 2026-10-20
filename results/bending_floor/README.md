# Strict Hermite bending precision study (#130)

This is an accuracy diagnosis, not a performance comparison. It uses the
96-DOF, two-polarization, 24-elements-per-polarization bending fixture from the
[FEM qualification](../fem_fast_block_v020/README.md). Six pairs are requested,
with a width-12 block and strict outer tolerance `1e-11`.

All six configurations below pass the shared `1e-8` physical spectrum,
residual, orthogonality and requested-subspace gates, but **only 4/6 meet the
requested strict tolerance** after 300 iterations. They remain rejected.

| Configuration | Worst normalized original residual |
|---|---:|
| [Public fast profile](baseline.json) | 9.044e-11 |
| [FP64 inner CG](fp64_cg.json) | 6.183e-11 |
| [No search bucketing](no_search.json) | 6.915e-11 |
| [No cached operator images or search bucketing](no_reuse.json) | 5.953e-11 |
| [Scalar-RHS outer sparse products](scalar_products.json) | 6.803e-11 |
| [No image reuse, search/CG buckets, or fused/symmetric/paired updates](basic.json) | 7.764e-11 |

Each occupancy JSON contains its exact executable command. Runs used the same
pinned environment and L40 as the preceding qualification. These controls show
that FP32 inner CG and the optimized block layout are not sufficient explanations
for the strict miss. They do not establish an absolute best achievable residual.

## Independent precision control

The [reference record](high_precision_reference.json) solves one polarization
using 60-decimal-digit arithmetic on the exact entries of the assembled FP64
matrices, then rounds the eigenpairs to FP64. Its first three distinct modes
are duplicated in the two-polarization problem.

| Reference | First-mode original residual after FP64 evaluation |
|---|---:|
| Dense generalized LAPACK | 2.860e-9 |
| LAPACK after stiffness-diagonal congruence, mapped back | 2.310e-9 |
| 60-digit solve, eigenpairs rounded to FP64 | 3.076e-11 |

Evaluating the rounded high-precision pair in extended precision still gives
`1.863e-11`. The stiffness and mass condition numbers are approximately
`8.00e6` and `4.66e5`. The LAPACK first eigenvalue differs from the high-precision
value by about `2.76e-10` relatively; the Warp first eigenvalue is much closer.
Thus the existing reference is adequate for the shared `1e-8` gates, but it
cannot certify an arbitrary `1e-11` original-residual target here.

This is evidence of sensitivity to FP64 representation and cancellation,
**not proof that every representable vector must exceed `1e-11`**. No production
algorithm or stopping tolerance was changed. The separately retained practical
Warp profile at `1e-8` passes all six modes. Higher precision or a different
formulation may improve stricter results; do not silently loosen caller tolerances.

Reproduce the offline reference (mpmath is an optional analysis dependency,
never part of the solver):

```bash
python -m pip install mpmath
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m \
  benchmarks.applications.bending_precision --elements 24 --digits 60 \
  --out build/bending-precision.json
```

The analysis script, matrices' hash, package versions and source hashes are
recorded. The solver remains pure Warp; all host reference work is outside it.
