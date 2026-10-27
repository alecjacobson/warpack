# Common application first-solve acceptance (#207)

2026-10-27, follow-up to the [completed library review](../README.md).
The numerical runtime is unchanged from merged main `33d4f26`. Changes are
confined to benchmark validation, tests and documentation. The existing full
suite, installed-wheel and matched-performance evidence remains at its recorded
source revision; these new small runs make no timing or speedup claim.

## Behavior

The common application harness immediately checks the first measured solve
using the same physical and native validator as later solves. It records
`first_solve_quality` and `first_solve_accepted`. A first-only failure rejects the
row while retaining later successful samples. The parent comparison also checks
the first evidence independently of the row's success flag. No failure is retried.

CG statistics are reset before each solve, including the first, and unfulfilled
inner solves reject acceptance. SVD records its core status per output. All
readbacks/reference calculations remain outside compute timers and outside
`warpack/`. Historical reports lacking first evidence are not retroactively
certified. Setup graph initialization, if used, remains distinct from first
measured use.

## Validation

- Pinned Warp 1.18/FSAI: **134 passed**, covering acceptance, common application
  first outputs, existing application tests, cuDSS provenance and CG tolerance
  regressions. [Log](pytest-final.log), [actual reports](pinned-reports.json).
- Stock Warp 1.15: **91 passed** for acceptance and first-output regressions.
  [Log](pytest-stock.log), [actual reports](stock-reports.json).
- Six first-output reporter regressions fail on the old harness, whose blob
  `1781a0d9d6ed36f130c8e976f8aa29ef35a9b243` is identical on baseline `a1b739a`
  and merged `33d4f26`. [Before-fix log](pytest-before-fix-actual.log).
- Lint and formatting pass. No core file changed, so this reporting-only
  follow-up does not require rerunning the full performance matrix.

The new fault-injection tests include wrong requested eigenvalues with small
physical residuals, incomplete native counts, nonzero status, and unfinished
inner solves only on the first output, followed by passing outputs. The SciPy
reference path is covered too. Real GPU cases check standard/generalized,
complex Hermitian, nonsymmetric, buckling and SVD outputs. Both cuDSS and pure
Warp CG pass a changed-start case under the documented setup graph-upload
protocol. CG uses the stricter `cg_tol=1e-14, cg_atol=0.0` for that qualification.

The **initial run failed**: 89 passed, 2 failed, retained in
[pytest-initial.log](pytest-initial.log). Diagnostic reruns are retained in
[pytest-diagnostic.log](pytest-diagnostic.log). The default inner CG tolerance
returned physically accurate vectors but only 3/4 native-converged eigenpairs;
relative-only tighter CG passes without changing the outer threshold
([check](pytest-relative-cg.log)). This is not a claim that the original settings
passed.

The other failure is an unresolved strict stopping floor in the small reciprocal
conformal fixture, [#209](https://github.com/alecjacobson/warpack/issues/209).
Three diagnostic reports retain both passing and rejected outputs:
[0](conformal-0.json), [1](conformal-1.json), [2](conformal-2.json). The second
has a warm residual of `1.1577e-11`, native count 1/2 at tolerance `1e-11`,
and passing physical accuracy. The final pinned report again rejects that
configuration. Its new test asserts honest rejection when native counts fail;
it **does not qualify that solver configuration** or relax its threshold.
A passing reporter test is distinct from an accepted numerical comparison.

## Remaining scope

Common-harness callers (including the direct FEM review worker) inherit this
change. Anvil direct and complete block comparison workers already check their
first outputs. Independent conformal, connection, reciprocal buckling,
transformed-pencil and experimental workers still need equivalent checks;
[#208](https://github.com/alecjacobson/warpack/issues/208) lists the audited sites.
The upstream graph-upload issue #61 also remains open. No historical first-use
claim or README timing row was changed.

## Reproduction

Use the pinned runtime from the library review, or stock Warp 1.15, on the same
L40/driver 570.158.01 and optional cuDSS 0.8.0.10. The focused pinned command is:

```sh
python -m pytest -q tests/test_benchmark_acceptance.py \
  tests/test_application_first_solve.py tests/test_applications.py \
  tests/test_cudss_provenance.py tests/test_cg_tolerances.py \
  tests/test_cg_absolute_tolerance.py
```

The stock command uses the first two test files. Source hashes are retained in
[source.json](source.json). Test reports include elapsed times for diagnostic
completeness; these small correctness checks are not performance comparisons.
