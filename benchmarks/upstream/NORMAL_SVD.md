# Explicit normal-product SVD / POD API

`warpack.NormalSVD` computes the largest resolved nonzero singular triplets of
a real FP64 compact scalar CSR matrix. It is useful for snapshot matrices whose
smaller dimension admits a modest search space. Numerical work is entirely Warp,
including sparse products, projected eigensolves, lifting, residuals and Gram
checks. cuDSS is not used. The existing `PartialSVD` augmented formulation remains
available with its original behavior.

```python
import warp as wp
from warpack import NormalSVD

# A: compact scalar FP64 Warp CSR, for example 100000 rows by 128 snapshots.
solver = NormalSVD(A, k=20, ncv=128, tol=1e-11, orthogonality_tol=1e-8)
graph = solver.capture(max_iterations=100)
wp.capture_launch(graph)
# Outputs remain on device: singular_values, row arrays u and v.
# Explicit host inspection, outside capture:
assert solver.status.numpy()[0] == 0
assert solver.count_left.numpy()[0] == solver.count_right.numpy()[0] == 20
```

The search operator is `A.T @ (A @ x)` for tall matrices and
`A @ (A.T @ x)` for wide ones. If `ncv` spans the entire smaller dimension
and is at most 64, the solver projects all coordinate vectors in one batch,
explicitly forming the small normal matrix. Larger or partial search spaces
use matrix-free Krylov expansion. The default basis-size formula is unchanged.
Sparse transposes and workspaces are allocated at setup. In-place value changes
are supported between replays at fixed sparse topology and array addresses;
initialization refreshes transpose values on device. Changing topology requires
new setup. Eager `solve(iterations=...)` uses the same finalization checks.

## Contract and limitations

- `1 <= k < min(A.shape)`; `k < ncv <= min(A.shape)` after clipping to the
  smaller dimension. The default `ncv` is `min(min(A.shape), max(4*k, 48))`.
- `singular_values` are descending. `u` has shape `(k, rows)` and `v` has shape
  `(k, columns)`, both FP64 row vectors.
- `left_residuals` and `right_residuals` measure the two original SVD equations,
  divided by `max(1, singular_value)`. Both counts must equal k at `tol`.
- `orthogonality_errors` stores the maximum entrywise deviations of `u u.T`
  and `v v.T` from identity. The default bound is 1e-8, independently of the
  equation tolerance. The scalar option must be finite and positive.
- Status 2 rejects nonfinite or unresolved requested normal Ritz values/lift
  norms: squared values must exceed 1e-14 times the largest requested normal
  Ritz value. Status 3 rejects nonfinite or nonorthogonal returned bases. Both
  clear accepted-triplet counts while retaining residual diagnostics. A prior
  conditioning failure takes precedence over the orthogonality failure.
- Status 0 alone is not convergence. Counts do not certify that a partial
  search space contains every requested mode. Partial search spaces retain the
  independent-chain exploration guard; a complete coordinate basis spans the
  whole normal space.

Normal products square conditioning. The cutoff is a conservative rejection
policy, not a rank estimate or an accuracy certificate. Requested zero singular
bases, complex matrices and weighted POD are unsupported by this API. Use the
augmented formulation when the conditioning tradeoff is unsuitable; this API
does not silently switch formulations or relax tolerances.

A full basis in a small snapshot dimension can avoid expensive restart
exploration and cover multiplicities. It also increases memory/setup costs;
allocating a full basis for a large smaller dimension is not generally practical.
Basis size is an explicit caller choice, not an automatic performance promise.
The automatic coordinate projection applies only when the selected basis is
already complete and has at most 64 vectors. It does not enlarge a partial basis.

## Evidence and implementation boundary

The [guard and wider-snapshot report](../experimental/NORMAL_GUARDS.md) records
39 focused passes on each tested Warp version, the rejected ill-conditioned
basis, restored-input capture regressions, and matched original-equation,
requested-space, orthogonality and reconstruction checks on 100,000×48 and
100,000×128 fixtures. The latter takes 74.955 ms with a full normal basis, versus
159.695 ms for native CuPy svds; its partial normal basis takes 434.726 ms and
is retained as a slower control. These initial guarded measurements precede
moving the unchanged numerical implementation into the installed package.

The implementation now lives in `warpack/normal_svd.py`. The comparison driver
and host reference fixture generation remain in `benchmarks/`. Installation
has no dependency on benchmark or application modules. This opt-in API addresses
the availability gap in [#31](https://github.com/alecjacobson/warpack/issues/31);
the original augmented/adaptive throughput regression remains tracked there.

Post-move controls reproduce the guarded timings: **15.753 ms** for
100,000×48 and **74.952 ms** for 100,000×128 with full bases, all three
trials passing and clean occupancy. [48-column control](../../results/performance_followup/normal-api-48.json),
[128-column control](../../results/performance_followup/normal-api-128.json).
The 128-column matrix hash matches the preceding all-method comparison.
An [isolated clean-source wheel check](../../results/performance_followup/normal-api-wheel.json)
imports the new API away from the source tree, with 23 core modules and no
benchmark modules installed.

The [combined local suite](../../results/performance_followup/normal-api-validation.json)
passes **710 tests in 348.92 s**, with lint and formatting checks passing.
Stock Warp 1.15 passes 39 focused captured/component cases plus two additional
public eager replay cases. This is local validation, not a new complete upstream
CuPy/Spectra conformance run.

The [small complete-basis follow-up](../experimental/FULL_NORMAL_BASIS.md) reduces
100,000×48 replay from **15.76 to 8.57 ms** and 100,000×64 from **27.21 to
14.58 ms**, with unchanged acceptance gates. Its 64-column native CuPy control
is **93.49 ms**. A 128-column coordinate experiment was slightly slower and is
kept outside automatic dispatch. Tall, wide, wave and changed-input regressions
exercise the installed implementation; conditioning safeguards still apply.
