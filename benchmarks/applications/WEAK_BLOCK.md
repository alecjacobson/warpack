# Recovering weak generalized modes

[Issue #46](https://github.com/alecjacobson/warpack/issues/46) exposed a block
orthogonalization failure, after cuDSS returned finite inverse images. The
64-unknown Hodge pencil contains a weak annulus, a strong torus, non-diagonal
SPD mass, three independent null directions and two requested weak positive
modes. At component scales `1e-10` and `1e-12`, the original two-pass metric
CholeskyQR returns non-finite modes. Small residuals would not establish that
all five requested directions were recovered.

The inverse image at scale `1e-10` has weighted condition number about
`4.2e11`. Its Gram matrix has a negative eigenvalue at roundoff scale, and even
host Cholesky fails. The sparse inverse output is finite; its largest relative
linear residual is `2.9e-5`. See the preserved
[intermediate diagnosis](../../results/weak_block_diagnosis.json) and
[shift-size probe](../../results/weak_block_probe.json). The probe timings include
first replay and are not steady performance measurements.

## Device-side recovery

`GeneralizedEigensolver` now retries a bad or very small Cholesky pivot before
applying its factor to the basis. The retry adds `1e-12 * trace(Gram)` to the
Gram diagonal. Two additional unshifted metric QR passes run only if a retry
occurred. The method follows the shifted-QR/QR2 stabilization principle of
[Fukaya et al., Shifted CholeskyQR](https://arxiv.org/abs/1809.11085), but our
trace-based shift and pivot trigger are practical heuristics, not the paper's
full sufficient bound for arbitrary weighted matrices.

The retry decision, factorization, triangular solves, mass products and
recovery count all stay in Warp on device. A captured conditional executes the
extra passes. Eager execution always runs those passes to avoid reading a
device flag on the host. `orthogonalization_recoveries` resets at initialization
and counts retries; `status` remains sticky on an unrecovered factor failure.
The shift changes only orthogonalization, not K, M or the physical target.
Rank-deficient or invalid blocks are not guaranteed recoverable.

The primary `GeneralizedKrylovSchur` path is unchanged. The generalized block,
block Cayley and block buckling paths inherit the safeguard.

## Accuracy and cost

The [same-budget comparison](../../results/weak_block_coverage.json) uses three
synchronized warm trials per profile on the L40, with no observed competing
GPU process. All methods use the same cuDSS factors, starts, ncv=32 and iteration
budget. Setup, capture, references and validation are outside replay timing.
The legacy profile overrides only orthogonalization with the pre-fix method.

| Fixture | Legacy QR2 | Conditional recovery | Always shifted QR + QR2 |
|---|---:|---:|---:|
| Weak scale `1e-4`, n=64, k=5, 100 steps | 82.743 ms | 87.827 ms | 121.638 ms |
| Weak scale `1e-10`, n=64, k=5, 100 steps | Rejected: non-finite | 88.652 ms | 121.358 ms |
| Weak scale `1e-12`, n=64, k=5, 100 steps | Rejected: non-finite | 88.740 ms | 121.618 ms |
| Two tori, n=10,086, k=6, 20 steps | 28.597 ms | 29.682 ms | 38.220 ms |

Every conditional and always-shifted trial passes physical residual, metric
orthogonality, complete requested-space and strict convergence-count checks.
Conditional recovery triggers once for each extreme weak solve and zero times
for the ordinary and large fixtures. The large-case overhead is about 3.8%,
versus about 34% for always-shifted QR. Rejected legacy timings remain in the
raw report and are not speed comparisons.

The weak positive reference divides the annulus stiffness by its small scale
*before* the independent dense generalized solve, then rescales the eigenvalues.
The known nullspace comes from incidence ranks/SVD. No reference vectors enter
solver starts. Zero eigenvalue errors are tested absolutely; positive errors
are normalized by the weak scale. This avoids using inaccurate tiny positive
values from a global dense solve spanning the full material contrast. The
requested-space projection also checks that a duplicate null direction cannot
substitute for a weak mode. This is eigenpair validation, not certified Betti
counting; the diagnostic cutoff can still classify the weak spectrum unresolved.

Graph DOT audits contain no host callbacks or DtoH copies. The same existing
cuDSS HtoD nodes remain (200 for 100 steps, 40 for 20 steps; #29). Recovery adds
no host transfer. DOT inspection is not a CUDA API synchronization trace.

The full local suite passes **292 tests in 161.81 seconds**, with no expected
failures. New regressions cover component scales `1e-4`, `1e-10`, `1e-12`,
repeated replay, device recovery flags changing without recapture, basis scales
from `1e-60` to `1e60`, invalid blocks and nested adaptive capture. Existing block
Cayley/buckling tests also pass. Lint, formatting and wheel build pass. See
[validation provenance](../../results/weak_block_validation.json).

The saved [10,890-DOF FEM block baseline](../../results/weak_block_fem_block.json)
passes accuracy and performance gates at 29.033 ms, a 1.0191× timing ratio. The
[10k A/B/C Krylov baselines](../../results/weak_block_existing_regression.json)
remain within 0.02% of their saved times; this does not resolve their separately
tracked tight internal-count limitations. The independent
[large Hodge baseline gate](../../results/weak_block_hodge_regression.json) also
passes, at 1.0362×. No Anvil or full upstream rerun is implied.

The [post-fix weak diagnostic](../../results/weak_block_nullspace_diagnostic.json)
now sees finite, independent pairs and reports flags 225: completeness
unverified, capacity exhausted, gap unresolved and expected-count mismatch.
It distinguishes two weak positive modes within five numerical near-zero modes.
Its solver counting gate intentionally remains false at the unchanged `1e-8`
cutoff; the independent component-scaled eigenpair gate above passes. Increasing
that cutoff or calling the observed numerical count a certified Betti number
would hide the distinction.

## Reproduction

Use the pinned Warp/FSAI environment described in [FEM.md](FEM.md). Run GPU jobs
sequentially:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest -q
python -m benchmarks.applications.weak_block_compare
python -m benchmarks.applications.weak_block_probe --out /tmp/weak-qr-probe.json
python -m benchmarks.applications.nullspace_compare --cases large_disconnected \
  --baseline results/hodge_large.json --out /tmp/weak-qr-hodge-regression.json
python -m benchmarks.applications.fem_compare --cases clamped_consistent \
  --side 10 --methods warp-block --ncv 32 --steps 10 \
  --baseline results/fem_side10_block32_cycles10.json --out /tmp/weak-qr-fem-block.json
```
