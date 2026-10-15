# Rig-constrained skinning eigenmodes

`ConstrainedKrylovSchur` implements the finite constrained weight modes in
[Fast Complementary Dynamics via Skinning Eigenmodes](https://arxiv.org/html/2303.11886v2)
(Benchekroun et al., SIGGRAPH 2023), Equation 22 and Appendix B.1.
Tracking: [#55](https://github.com/alecjacobson/warpack/issues/55),
under [#25](https://github.com/alecjacobson/warpack/issues/25).

## Input and numerical contract

For n vertices and r independent constraints:

```
H_w W + C.T mu = M_w W Lambda
C W = 0
W.T M_w W = I
```

H_w is the real symmetric n-by-n **scalar weight** stiffness, not the 3n-by-3n
displacement stiffness. `warpack.skinning.skinning_stiffness(H)` computes the
sum of the three coordinate diagonal blocks of an interleaved 3x3-block elastic
Hessian. M_w is the scalar SPD vertex mass; consistent mass is also supported
by the solver. Each weight can generate twelve affine displacement coordinates.

`skinning_constraint_rows(X, J_effective)` implements the coordinate-weighted
rig products in Appendix B.1. `J_effective` has shape `(parameters,n,3)` and must
already include the desired mass and momentum-leakage weighting (D M J in the
paper). Each parameter contributes x/y/z displacement components times
`[X,Y,Z,1]`. This generally produces redundant rows.

`independent_constraints(rows)` normalizes nonzero input rows and performs
pivoted, twice-reorthogonalized row QR in Warp. It returns orthonormal constraint
rows as scalar BSR, dropping residual directions below `tolerance` (default
1e-10). Verify approximate rank choices against the original rig rows. This
O(p²n) preprocessing targets small rig spaces, not huge constraint systems.
Nonfinite inputs, rank zero, and constraints spanning all physical coordinates
are rejected. Callers with an already independent sparse C can skip it.

The shifted solve is

```
[ H_w - sigma M_w   C.T ] [ z  ] = [ M_w x ]
[ C                 0  ] [ nu ]   [   0   ]
```

The physical operator `T x = z` is self-adjoint in the SPD M_w inner product.
Its nonzero eigenvalues are `1/(lambda-sigma)` for the finite constrained modes;
its zero spectrum represents the excluded infinite part. We target largest
transformed magnitude, hence finite eigenvalues nearest sigma. There is no
dense nullspace basis in the production solve and no singular-mass inner product.
The augmented mass is never passed to `GeneralizedKrylovSchur`.

H_w need not be positive definite, but the KKT matrix must be nonsingular:
constraints must be independent and sigma must avoid the constrained spectrum.
The API requires `1 <= k < n-r` and `k < ncv <= n`. Residual convergence alone
is not a proof of global spectrum completeness, especially at multiplicities.

The returned multiplier for each Ritz vector uses the same KKT factors:
`mu = (lambda-sigma) nu`. Acceptance checks the original equilibrium including
constraint forces and separately checks every constraint row:

```
residuals = ||H_w w + C.T mu - lambda M_w w|| /
            max(1, ||H_w w|| + ||C.T mu|| + |lambda| ||M_w w||)
constraint_residuals = max_i |C_i w| / (||C_i|| ||w||)
```

Both must meet `tol` to increment device `converged`. The unconstrained residual
`H_w w - lambda M_w w` is generally nonzero and is not the correct criterion.
`MassOrthogonalComplementInverse` remains the separate invariant-space API;
simply projecting an unconstrained inverse does not solve this problem.

## Use and graph lifecycle

```python
import warp as wp

from warpack import ConstrainedKrylovSchur, CuDSSInverse, SparseOperator, saddle_point_matrix
from warpack.skinning import independent_constraints, skinning_constraint_rows

# H_w and M_w: compressed scalar FP64 Warp BSR; X: vec3d array.
raw = skinning_constraint_rows(X, J_effective)
C = independent_constraints(raw)
sigma = -0.1
KKT = saddle_point_matrix(H_w, M_w, C, shift=sigma)
inverse = CuDSSInverse(
    KKT, 1, mtype="general", matching=True, refinement_steps=1, cooperative_refinement=True
)
solver = ConstrainedKrylovSchur(
    SparseOperator(H_w),
    SparseOperator(M_w),
    C,
    inverse,
    k=20,
    ncv=64,
    target=sigma,
    tol=1e-9,
)
graph = solver.capture(max_iterations=2)  # choose/validate a budget for your problem
wp.capture_launch(graph)
# Device outputs: eigenvalues, eigenvectors (20,n), multipliers (20,r),
# residuals, constraint_residuals, converged.
# Release graph references before inverse.close().
```

All numerical preprocessing and iteration in these APIs is Warp, except cuDSS
factorization/solves. Setup reads back **integer rank/validity metadata** to allocate the
constraint matrix and synchronizes sparse structure counts; it is not captured.
Factor setup, JIT, and capture are separate from replay. Iteration, mass products,
orthogonalization, projected solves, multipliers and convergence tests stay on
device. Update the optional device initial vector between replays without
recapture. Rebuild constraints/factors/solver after changing the rig or matrices.
Instances own scratch buffers and cannot run concurrently on different streams.

cuDSS requires a fixed outer graph budget. Captured graphs still contain vendor
scalar HtoD copies, tracked in [#29](https://github.com/alecjacobson/warpack/issues/29).
Graph capturability does **not** mean that those copies have been eliminated.
The pure-Warp test inverse also exercises adaptive capture without host transfers.

## Tests and comparisons

`tests/test_constrained.py` includes independent dense nullspace/LAPACK references,
non-invariant constraints, lumped/consistent mass, interior targeting, repeated
eigenvalues, admissible zero modes, infeasible initial vectors, changed RHS/start
replay, constraint-gated rejection of otherwise exact eigenvectors, and redundant
rig-row reduction. A tetrahedral application test independently assembles elastic
stiffness and checks the appendix matrix construction and complete requested space.

The benchmark uses a heterogeneous tetrahedral 3:1:1 beam and two localized angular
rig parameters. The 24 raw coordinate-weighted rows reduce to 14 independent rows.
All methods solve the same constrained problem. CPU/CuPy use the symmetric
mass-scaled physical operator `sqrt(M) [I,0] KKT^-1 [I;0] sqrt(M)`; Warp uses the
equivalent physical M metric. A dense eliminated reference is used up to 400
vertices; larger references use a different seed, larger Krylov space, tighter
tolerance and original-equation validation.

- SciPy: ARPACK with SuperLU KKT factors.
- Spectra: native C++ SymEigsSolver with Eigen SparseLU KKT factors.
- CuPy+cuDSS: CuPy eigsh on an explicitly supplied transformed LinearOperator;
  cuDSS comes from **our adapter**, not a native CuPy eigsh backend.
- Warp+cuDSS: captured ConstrainedKrylovSchur, matched general factors with one
  Warp residual correction, including multiplier and physical residual evaluation.
  The final GPU profiles enable `cooperative_refinement=True` on both backends:
  a warp cooperates on each correction-product row, avoiding serial traversal
  of the long rig rows. The default thread-per-row path is unchanged.
- Warp+CG: explicitly unsupported on the indefinite KKT matrix. A constrained
  iterative inverse is tracked alongside [#49](https://github.com/alecjacobson/warpack/issues/49).

Every warmup and timed trial must pass equilibrium and constraint residuals,
requested eigenvalues, mass orthogonality and complete requested subspace checks
at 1e-7. Failed timings remain diagnostics. Native reference tolerances are 1e-11;
Warp reports its separate 1e-9 strict convergence count. Timings separate setup,
capture/warmup and three synchronized solve samples. Reference/input construction
and host validation are outside timing; Warp includes its device validation in
replay. CPU setup covers KKT factors; GPU setup also includes transfers and Warp
KKT assembly. Spectra compilation is excluded. Initial JIT can affect setup on
a cold cache, and CPU contention is not isolated by this harness.

## Recorded results (2026-10-15)

NVIDIA L40, Warp 1.18.0 with the pinned FSAI overlay, cuDSS 0.8.0.10,
CuPy 14.2.0, SciPy 1.15.0; Intel Xeon Platinum 8362 with four CPU BLAS threads.
The native comparison uses Eigen 3.4.0, GCC 11.4.0 (`-O3 -DNDEBUG`), and
Spectra revision `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`. The large fixture has
**45,625 vertices, 644,121 stiffness nonzeros, 1,857,313 KKT nonzeros,
14 independent constraints, 20 requested modes, ncv=64, sigma=-0.1**.
Warp uses two restart cycles and converges 20/20 at its stricter 1e-9 tolerance.

CPU rows are from the [initial full comparison](../../results/skinning_large.json).
GPU rows are from the [confirmed cooperative-correction comparison](../../results/skinning_large_confirmed.json).
CPU-heavy `geometric_peel` work was observed during the original large run;
CPU timing is not an isolated machine comparison. No competing GPU process
was reported at benchmark startup. GPU timing also is not a continuous contention trace.

| Backend | Setup (s) | Median solve/replay (s) | Worst equilibrium residual |
|---|---:|---:|---:|
| ARPACK + SuperLU | 26.849 | 11.988 | 1.69e-12 |
| Spectra + Eigen SparseLU | 13.876 | 8.114 | 1.69e-12 |
| CuPy eigsh + supplied cuDSS inverse | 0.885 | 0.290 | 1.72e-12 |
| Warp + cuDSS, two cycles | 0.702 | 0.339 | 4.00e-13 |

The initial thread-per-row correction cost **1.229 s** in Warp and **1.056 s**
in CuPy. A [repeat](../../results/skinning_large_regression.json) reproduced both
within 0.2%. [Cooperative correction](../../results/skinning_large_cooperative.json)
cut both to about 27.5% of their previous times, with no accuracy regression.
The final repeat checks the same problem/device/runtime/configuration and
passes the 25% timing-regression gate. This improves the common inverse;
the remaining roughly 17% Warp/CuPy gap is [#56](https://github.com/alecjacobson/warpack/issues/56).
Warp's physical validation includes 20 extra KKT applications to recover
multipliers, which is one candidate for further optimization.

The [2,025-vertex budget probe](../../results/skinning_medium_two_cycles.json)
also passes with two cycles (0.082 s before cooperative correction), compared
with [eight cycles](../../results/skinning_medium.json) at 0.256 s.
The faster [one-cycle result](../../results/skinning_medium_one_cycle.json)
is **rejected**: eigenvalue error 6.59e-9 looks acceptable, but equilibrium
residual 6.53e-4 and constraint error 3.57e-7 fail the physical gates; only
11/20 meet the device tolerance. This is not a speedup result.

The large two-cycle graph has **zero host callback nodes** and **500 vendor
scalar HtoD copy nodes**, plus device kernels/copies/memsets and Warp recovery
conditionals. Those transfer nodes remain a separate unresolved limitation (#29).

The full-suite rerun also exposed an intermittent pre-existing **Hodge** strict
count at a two-cycle budget (5/6 at 1e-11, with broader accuracy checks passing).
The [preserved focused probe](../../results/hodge_strict_probe.json) reproduces
that boundary and passes all 20 replays at three and four cycles, with and without
inverse correction. The direct Hodge regression now uses three cycles with all
original tolerances/assertions retained; CG budgets and production solver defaults
are unchanged. See [#57](https://github.com/alecjacobson/warpack/issues/57).

The stock Warp 1.15 check exposed and fixed a setup bug in the new weight
stiffness builder: after replacing sparse topology it must call
`notify_nnz_changed`, otherwise a cached empty-matrix count can survive into
KKT assembly even though direct stiffness products are correct. The regression
checks the synchronized structure count and final modes; all twelve constrained
tests now pass on stock 1.15. Evidence and fix: [#58](https://github.com/alecjacobson/warpack/issues/58).

Reproduction requires Eigen headers, a C++17 compiler, and the
[pinned Spectra checkout](../upstream/README.md#reproduce) at
`build/upstream-spectra`; the skinning harness compiles its native adapter
automatically. The source revision used above is recorded explicitly.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest tests/test_constrained.py -q
python -m benchmarks.applications.skinning_compare --side 8 --steps 2 \
  --out results/skinning_medium_baseline.json
python -m benchmarks.applications.skinning_compare --side 8 --steps 2 \
  --baseline results/skinning_medium_baseline.json --out results/skinning_medium_regression.json
python -m benchmarks.applications.skinning_compare --side 24 --steps 2 \
  --out results/skinning_large.json
```

This adds the constrained saddle-point subclass, **not** an arbitrary singular-mass
eigensolver. The broader spectral-conformal/singular-pencil issue remains open.
