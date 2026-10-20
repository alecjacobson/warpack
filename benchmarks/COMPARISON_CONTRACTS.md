# Composite comparison process acceptance

Fix for [#85](https://github.com/alecjacobson/warpack/issues/85), following the
block/related-start driver repair in #84. A passing primary solver previously
could hide a failed requested reference behind an overall exit status of zero.
Some application drivers only enforced acceptance when comparing a baseline;
others silently permitted explicitly requested unsupported backends.

The default parent comparison now requires every requested row to pass its
recorded physical and stopping gates. Unsupported, error, timeout and worker
failure rows reject the run. Empty reports and success flags without trial
evidence are rejected. Recorded full counts, status codes, native stopping
bounds and inverse-CG failures are checked without inventing new thresholds.
Numerical iteration, residual definitions and tolerances are unchanged.

Each row retains its existing `accepted`/`outcome` and trial data. New
`numerical_contract_accepted` and `numerical_contract_failures` fields state the
combined result; the report has its own `numerical_contract_accepted`. The
complete report is written before enforcing the default nonzero exit. Existing
graph-compliance, occupancy and performance-baseline fields stay separate;
passing a numerical check does not establish timing or graph eligibility.

`--diagnostic` explicitly permits rejected requested methods without failing
the numerical process gate. The report still records `diagnostic: true` and
`numerical_contract_accepted: false`; failures never become passes. Existing
timing and baseline-regression gates still apply. This is useful for an
inventory that deliberately includes unsupported interfaces or known failures.
Internal `--worker` subprocesses transport their result rows to the parent;
they are not standalone comparison acceptance commands.

## Audited entry points

- `benchmarks.experimental.normal_pod`: both the primary normal solver and
  all requested reference methods are mandatory. The acceptance-helper source
  hash is included in the report.
- `benchmarks.applications.compare`, `fem_compare`, `hodge_compare`: enforce
  recorded numerical contracts even without a baseline. The general driver
  also removes a previous worker-result file before launching that worker.
- `interior_compare`, `negative_compare`, `skinning_compare`: explicitly
  requested unsupported methods no longer silently qualify as a passing run.
- `pod_compare`: uses the same contract and diagnostic policy while retaining
  its occupancy gate and strict-count checks.

The block/related-start drivers retain #84's specialized full-block stopping
gates. Existing nullspace diagnostics and deliberately failed recovery-control
experiments retain their application-specific expected-outcome gates. This
change does not reinterpret upstream conformance inventories as numerical
benchmark passes, or rewrite historical raw results.

## Verification

**30 focused tests pass in .81s**, covering physical failures, incomplete
counts, native tolerance misses, status errors, inverse failures, empty/missing
evidence, unsupported/error/timeout rows, worker failures, diagnostic mode,
retained row contents and separation from graph/timing results. Ten parent-CLI
cases exercise successful, rejected and diagnostic worker transport. All seven
application entry points expose the diagnostic option. Reassessing saved
regular-FEM records accepts the verified 12-cycle profile and rejects the
known 8-cycle failures without modifying either original artifact.

Real sequential CLI checks use a small 1000×64 POD fixture and an explicitly
unsupported general-eigenproblem request:

| Requested comparison | Default exit | Diagnostic exit | Numerical report |
|---|---:|---:|---|
| Warp passes; SciPy receives invalid reference `ncv=20` for `k=20` | 1 | 0 | rejected |
| Warp and valid SciPy reference `ncv=48` pass | 0 | — | accepted |
| Class H general problem, requested native CuPy backend unsupported | 1 | 0 | rejected |

The invalid reference width is an intentional process-contract control, not a
new ARPACK numerical bug. Both its failed reference row and the accepted Warp
row are preserved. These single-sample checks validate CLI behavior; they are
not performance comparisons.

[Rejected reference](../results/performance_followup/contract-normal-reference-error-final.json),
[diagnostic reference failure](../results/performance_followup/contract-normal-diagnostic-final.json),
[accepted comparison](../results/performance_followup/contract-normal-accepted-final.json),
[unsupported request](../results/performance_followup/contract-unsupported-final.json),
[diagnostic unsupported request](../results/performance_followup/contract-unsupported-diagnostic-final.json).
[Validation and exact commands](../results/performance_followup/comparison-contract-validation.json)
record the helper/driver hashes, test scope and audit results. No production
solver code changes in this follow-up; the preceding numerical validation
remains applicable.
