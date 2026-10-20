# Generalized regular coverage and per-replay acceptance

All **27 newly enabled LM/LA/SA Spectra cases pass**, with the original fixture
widths, budgets and tolerances. Across the complete `SymGEigsCholesky` and
`SymGEigsRegInv` families, **36 cases pass and nine BothEnds cases remain
unsupported**. All nine previously supported SM cases retain their passes.
Every accepted case passes both independently validated graph replays.

The existing public `GeneralizedKrylovSchur(mode="regular", which=...)` was
already tested in the [regular FEM comparisons](../applications/REGULAR_FEM.md),
but the conformance adapter still skipped these selections. It now supplies
a cuDSS inverse of B to the B-metric regular method. The original A-inverse SM
route is unchanged. This solves the same generalized pencil; it does not claim
to implement Spectra's Cholesky algorithm. Production numerical code and
performance defaults are unchanged; no timing improvement is claimed.

## Stronger independent checks

The adapter now checks the requested spectrum, B-orthogonality and full allowed
subspace against an independent dense generalized reference for these two
SPD-mass families. The projector uses the B-inner product. Boundary ties permit
any valid boundary member, but injective assignment and mandatory-mode coverage
reject omission of a strictly required mode. Exact residuals from the wrong
spectrum cannot pass these checks.

Worst errors across all36 accepted cases and both replays:

| Check | Maximum error |
|---|---:|
| Original absolute residual | 3.350e-13 |
| Requested eigenvalue assignment | 2.154e-14 |
| B-orthogonality | 6.773e-14 |
| Allowed-subspace error | 2.671e-12 |

Every device status is zero and every required count is complete. The unchanged
absolute acceptance bound is1e-9. Unsupported BothEnds rows have no invented
numerical results.

## Every replay is now checked

The prior accuracy-only runner launched twice and inspected only the final
output. That limitation remains explicit in the [frozen full baseline](SPECTRA_REFRESH.md).
The runner now validates after each replay and retains `accuracy_trials`.
Aggregate success requires every trial to pass; summary errors are maxima,
counts are minima, and any failed/missing status remains rejecting. Warmup is
separate. References and host inspection remain outside captured computation.

Sixty-four focused tests pass on pinned Warp1.18 and stock1.15. Thirteen new
CPU checks cover metric selection, ties/multiplicity and first-fail/last-pass
aggregation (including SVD's two physical counts). Six new actual GPU adapter
cases cover indefinite A/non-diagonal SPD B, all three regular selections,
both families, and both independently checked replays. Lint/format checks pass.

Overlaying these45 cases onto the frozen360-case baseline gives **197 passes,
14 failures and149 unsupported**, with no lost passes or checked-numerical
regressions. This is a **composite inventory**, not a new full per-replay run;
the remaining315 rows retain the baseline's final-replay-only protocol.

## Evidence and reproduction

- [All45 original family cases, including per-replay results](../../results/performance_followup/spectra-regular-followup.json)
- [Validation, exact source hashes and worst errors](../../results/performance_followup/spectra-regular-validation.json)
- [Explicit composite comparison](../../results/performance_followup/spectra-regular-comparison.json)
- [Regenerable fixture hashes](../../results/performance_followup/spectra-regular-fixture-hashes.json)

```bash
python -m benchmarks.upstream.prepare_spectra \
  --families SymGEigsCholesky SymGEigsRegInv --out build/regular-fixtures
python -m benchmarks.upstream.bounded spectra --fixtures build/regular-fixtures \
  --families SymGEigsCholesky,SymGEigsRegInv --accuracy-only --timeout 120 \
  --out build/regular-followup.json
```

Use the pinned environment and serialized GPU execution. The recorded candidate
is `bae39e5`; the later integration also includes the independently validated
GMRES restart setup option, without changing this default replay arithmetic.
