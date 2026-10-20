# General smallest-magnitude convergence, 2026-10-17

All five numerical failures listed in [#12](https://github.com/alecjacobson/warpack/issues/12)
pass a documented convergence profile: **matched cuDSS factors with one FP64
correction, 200 outer cycles for the compact n=10/k=3/ncv=6 cases, and the
original 30 cycles for n=100/k=10/ncv=30**. No tolerance or matrix is changed.
This establishes achievable accuracy; it does not fix all cases within the old
30-cycle budget or demonstrate a speedup.

The full small general-SM selection has **eight passes**, including three
additional controls. Worst original residual is **6.42e-15**, requested-spectrum
error **1.47e-14**, and invariant-subspace error **1.50e-14**. All strict physical
convergence counts reach k at the original internal tolerance of 1e-12.
[Validated profile](../../results/general_sm_validated_profile.json) and
[comparison](../../results/general_sm_profile_comparison.json).

## Separate causes

The [complex matching fix](HERMITIAN_SHIFT.md) repairs inaccurate sparse inverse
factors. With matching but the old 30-cycle budget, three of #12's five cases
pass: `GenEigs_21`, `GenEigs_27`, and `ComplexEigs_27`. The remaining small cases
need more Arnoldi cycles. A further control, `ComplexEigs_3`, passes the upstream
residual bound but initially reports only 2/3 at the stricter internal tolerance.
[Matched, original-budget report](../../results/general_sm_small_after_matching.json).

The preserved [budget controls](../../results/general_sm_budget_controls.json)
use the original restart and unchanged k/ncv:

| Case | Cycles | Max original residual | Converged / k |
|---|---:|---:|---:|
| GenEigs_3 | 30 | 4.58e-4 | 1/3 |
| GenEigs_3 | 60 | 1.41e-6 | 1/3 |
| GenEigs_3 | 100 | 6.14e-10 | 1/3 |
| GenEigs_3 | 200 | 6.42e-15 | 3/3 |
| ComplexEigs_21 | 30 | 3.96e-6 | 1/3 |
| ComplexEigs_21 | 60 | 1.84e-10 | 2/3 |
| ComplexEigs_21 | 100 | 7.13e-16 | 3/3 |
| ComplexEigs_3 | 30 | 3.00e-12 | 2/3 |
| ComplexEigs_3 | 60 | 1.64e-15 | 3/3 |

The 100-cycle `GenEigs_3` result meets the original residual assertion but still
has a short strict count; it remains a failed strict result. Upstream permits
nonconvergence in these SM sections. The successful profile uses 200 cycles for
all compact cases, rather than claiming that the old default already passed.

Three restart experiments were rejected and **are not in production**:

- [Retain three of six vectors](../../results/general_sm_small_after_restart.json):
  improved the sparse complex case but worsened the dense real case.
- [Retain four of six](../../results/general_sm_small_balanced_restart.json):
  helped one count case but left the two numerical failures.
- [Expand from the worst requested Ritz residual](../../results/general_sm_small_residual_restart.json):
  essentially unchanged outcomes with extra reductions/rotations.

The retained-count controls use `min(k+4,max(k,ncv//2))` and
`min(k+4,max(k,(ncv+k)//2))` for extremal selection. The residual control keeps
the original count, rotates cached operator products to compute relative Ritz
residuals, and swaps the worst requested vector to the last retained row before
orthogonalization. Their JSON reports identify them as rejected experiments.
At this historical checkpoint the shipped Arnoldi implementation remained
unchanged from `bc2db07`. The later [general restart fix](../experimental/GENERAL_RESTART.md)
uses the actual requested residual as a new direction, rather than swapping
Ritz-vector order, and validates the two larger strict-count cases in #35.

## Coverage and API implications

The Spectra runner now checks SM selection independently for general real and
complex problems. It uses an injective eigenvalue assignment, accepts either
member of a conjugate pair when k cuts an equal-magnitude boundary, checks
vector norms, and checks membership in the requested right invariant subspace.
It does not require nonsymmetric eigenvectors to be mutually orthogonal.

The default runner budgets remain unchanged. `--iterations` is an explicit
override, recorded in the output. `general_sm_accuracy` provides the validated
compact/wider profile and filters out matrices larger than 100 unknowns.
No setup/solve timing or large-problem run was made for this follow-up.

Six local fixture regressions independently check the original inverse solve,
eigenvalues, invariant space, normalization, status and full physical counts on
two captured replays. They preserve the five #12 matrices plus `ComplexEigs_3`
in a 32 KiB fixture. **49 related local tests pass**, including general Arnoldi,
Schur, real/complex shift-invert and reporting contracts. Source provenance is
in [the fixture notes](../../tests/fixtures/README.md).
The six new fixture regressions also pass on stock Warp 1.15. Repository lint
and changed-file formatting pass. This follow-up changes tests, reporting and
the explicit profile runner; it does not modify production solver code.

The existing cuDSS graph-copy issue #29 and general-solver performance issue #14
remain open. The larger strict-count cases in #35 were not rerun. The successful
200-cycle setting is a convergence budget, not a claim that more cycles are free
or that compact Arnoldi performance is solved. Anvil timing tables are unchanged.

## Reproduction

Use the pinned Warp/CuPy/Spectra setup in [README.md](README.md):

```bash
python -m benchmarks.upstream.prepare_spectra --families GenEigs ComplexEigs \
  --max-size 100 --out build/general-sm-followup/fixtures
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.upstream.general_sm_accuracy \
  --out build/general-sm-followup/validated.json
python -m benchmarks.upstream.run_spectra --selection SM \
  --fixtures build/general-sm-followup/fixtures --accuracy-only \
  --out build/general-sm-followup/original-budget.json
# A single budget-control point; use 30/60/100/200 and the cases listed above.
python -m benchmarks.upstream.run_spectra --case GenEigs_3 --iterations 200 \
  --fixtures build/general-sm-followup/fixtures --accuracy-only \
  --out build/general-sm-followup/control.json
python -m pytest -q tests/test_general_sm.py tests/test_arnoldi.py \
  tests/test_schur.py tests/test_shift.py tests/test_upstream_contracts.py
```
