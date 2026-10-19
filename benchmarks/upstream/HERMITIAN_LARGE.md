# Complete Spectra Hermitian family follow-up

On 2026-10-19, **all 30 exported Hermitian fixtures pass**, including the ten
1,000-complex-unknown cases left open in
[#9](https://github.com/alecjacobson/warpack/issues/9). This completes that issue's
numerical acceptance. It does not establish generalized complex-mass support
or close the broader connection-Laplacian coverage issue #24.

No production arithmetic changed in this follow-up. The previously merged
[realified Krylov solver](HERMITIAN.md) and
[matched cuDSS Hermitian shift-invert path](HERMITIAN_SHIFT.md) resolve the
original failures at full size as well as the smaller fixtures.

## Accuracy and actual work

Exact pinned Spectra matrices, k, selections and requested subspace widths;
unchanged original absolute tolerance 1e-9 and internal normalized tolerance
1e-12. Every case passes the original residual bound, full convergence counts,
independent requested-spectrum comparison, complex conjugate orthogonality,
and invariant-subspace checks. The complete family was captured and replayed
twice per case; the adapter validates the final replay. Separate local graph
regressions validate changed inputs on each replay.

| Complex unknowns | Cases | Worst absolute residual | Worst eigenvalue error | Worst conjugate orthogonality error | Worst subspace error |
|---|---:|---:|---:|---:|---:|
| 10 | 10 | 1.087e-14 | 3.553e-14 | 4.443e-16 | 1.082e-14 |
| 100 | 10 | 1.767e-13 | 5.472e-13 | 7.772e-16 | 6.091e-14 |
| 1000 | 10 | 7.173e-13 | 8.655e-12 | 4.441e-16 | 7.277e-12 |

Each size has dense and sparse matrices with LM/LA/SM/SA/BE selection. Direct
LM/LA/SA/BE use pure-Warp Hermitian Krylov on a realification. SM uses matched
cuDSS factors, one fixed Warp residual correction, and Hermitian inverse
iteration; the complex original equation is checked after unshifting. There
is no CG approximation in this family. The direct budget remains 300 restart
cycles, and the shifted budget remains 30 fixed cycles. Direct graphs stop
on device; the reports now include executed `outer_iterations`.

The native unmodified Spectra Hermitian executable also passes: **60 assertions
in six Catch test cases** (30 selection sections). The original Warpack report
had 14 passes and 16 strict failures in this family. The current run loses no
previous numerical or strict pass. Regenerated small fixture exports are
bitwise identical to the earlier 20-case export; changing the exporter size
limit does not change the RNG or case numbering.

[Full rows](../../results/performance_followup/hermitian-full.json) ·
[Contract comparison](../../results/performance_followup/hermitian-contracts.json) ·
[Validation and fixture provenance](../../results/performance_followup/hermitian-validation.json).

The first full-size baseline also passes all ten cases and includes single-trial
setup/capture/replay diagnostics. Those are **not accepted performance comparisons**:
there are no matched repeated native timing controls here. They are retained in
`hermitian-large-baseline.json`, with its historical width-reporting semantics.
The final 30-case result is accuracy-only. Anvil timings are unaffected.

## Corrected basis metadata

This rerun exposed [#65](https://github.com/alecjacobson/warpack/issues/65): the
adapter called the requested fixture width `effective_ncv`, even when the
constructed solver doubled or clamped it. For example, n=1000/k=20/ncv50 uses
**100 real basis vectors of length 2000**, not 50 complex vectors internally.
This bug affects reporting, not solver arithmetic or the selected settings.

New rows separate `requested_ncv`, actual allocated `effective_ncv`,
`basis_vector_length`, `basis_scalar_type`, and `basis_solver_type`. Widths come
from the constructed solver, including shifted Hermitian and SVD wrappers,
clamping and defaults. Executed cycles are read after solving, outside timing
and capture. Historical JSON is preserved, not relabeled retroactively.

Five regression cases verify these fields against real constructed solvers.
**78 focused tests pass** across reporting contracts, Hermitian recovery and
shifted solves. Lint and formatting pass. The preceding production validation
remains **500 local tests**; this metadata-only follow-up is a focused rerun,
not another full-suite run. Existing multiplicity, complex-start, changed-value
and graph audits remain applicable. The cuDSS vendor-copy limitation stays #29.

## Reproduce

Use the pinned environment and upstream checkout from [README.md](README.md).
The exports are local build artifacts; no large fixture binaries are added to
the repository.

```bash
python -m benchmarks.upstream.prepare_spectra --families HermEigs \
  --out build/hermitian-large/fixtures
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.upstream.run_spectra --family HermEigs \
  --fixtures build/hermitian-large/fixtures --accuracy-only \
  --out build/hermitian-full.json
build/upstream-spectra-build/test/HermEigs
python -m pytest -q tests/test_upstream_contracts.py tests/test_hermitian.py \
  tests/test_shift.py
```
