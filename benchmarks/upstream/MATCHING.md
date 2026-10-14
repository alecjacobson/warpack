# Matched cuDSS factors for sparse indefinite problems

The sparse SM failures in [#4](https://github.com/alecjacobson/warpack/issues/4)
are primarily inverse-factor accuracy failures. Their exact Spectra matrices
have condition numbers about 17, 134 and 5,480, but mostly zero diagonals.
One correction of the original symmetric factors does not meet the upstream
residual bound on these fixtures. Increasing that correction budget
alone is expensive and does not reliably recover the requested accuracy.

`CuDSSInverse(A, width, mtype="general", matching=True)` enables cuDSS's
maximum-diagonal-product column matching and scaling before factorization,
with its default ordering. This is an explicit factorization choice: neither
the matrix nor its requested spectrum is changed. The solve undoes cuDSS's
permutations and scaling. Matching requires general factors; it cannot be
combined with COLAMD/BTF_COLAMD. These settings and restrictions follow the
[cuDSS configuration documentation](https://docs.nvidia.com/cuda/cudss/types.html).

The public default remains SPD with zero correction. Legacy symmetric/general
factors without matching retain one Warp correction. Explicit matching defaults
to zero correction; `refinement_steps=1` adds one fixed correction when needed.
The unshifted real SM and generalized Spectra inverse adapters use
**matching plus one correction** at their strict internal tolerance of `1e-12`.
Standard real shifted families retain their established symmetric profile. The CuPy real inverse adapter and
interior/buckling application benchmarks use **matching with zero correction**.
These profile choices are recorded separately from outer-cycle budgets.

Matching is cuDSS setup work, outside capture. Correction and eigensolver
arithmetic remain Warp; cuDSS applies its factors. Matching does not remove the
known vendor HtoD scalar-copy nodes, so [#29](https://github.com/alecjacobson/warpack/issues/29)
remains open. No host residual decision or host callback is added to replay.

## Why retain an explicit correction option?

The initial matching-only diagnostic passed all three original spectra.
However, the first complete local suite then returned **19/20 inverse pairs**
on a replay of `SymEigs_27`, even though the physical count was 20/20 and the
spectrum, residual, orthogonality and subspace assertions passed. That run had
189 passes and one failure; it is not counted as successful validation.
The permanent strict-contract regression and Spectra adapter therefore use
matching plus one correction. The assertions, `k`, `ncv`, 100 outer cycles and
original residual bounds are unchanged. This is an explicit inner-solve budget
change, not a relaxation of convergence reporting.

The diagnostic factor sweeps are retained:

- [Original symmetric factors, 0–3 corrections](../../results/refinement_symmetric_probe.json).
- [Original general factors](../../results/refinement_general_probe.json).
- [Matching linear residuals](../../results/pivot_matching_linear.json) and
  [matching eigenproblems](../../results/pivot_matching_eigen.json).
- [Matching application profile, zero correction](../../results/pivot_matching_applications_0.json)
  and [one correction](../../results/pivot_matching_applications_1.json).

These initial probes used a local wrapper configuration hook before the public
API existed. Their configuration fields describe the actual settings. The
public reproduction command below supplies the same options directly. One
successful factorization/replay is not a guarantee over every later replay.

## Rejected global-pivoting policy

Global pivoting (`mtype="general", reordering="colamd"`) also solves all three
sparse SM fixtures accurately; permanent regressions cover that explicit option.
It is not the default. An exploratory automatic COLAMD policy on symmetric
factors made the interior benchmark about 5.9 times slower at n=1,024 and
41 times slower at n=10,000 relative to the pre-correction baseline. Two of
three large trials also missed the `1e-8` residual gate. Those timings are
failed diagnostics, not accepted performance results.

That experimental symmetric/COLAMD combination was accepted by the installed
runtime, but the documentation supports COLAMD for general matrix factors.
The public examples and permanent tests use the documented general route.
The automatic policy was reverted before the implementation commit.

[General COLAMD probe](../../results/pivot_colamd_general_probe.json),
[experimental symmetric COLAMD probe](../../results/pivot_colamd_symmetric_probe.json),
[small application results](../../results/pivot_interior_small.json), and
[large failed application results](../../results/pivot_interior_large.json)
preserve the rejected experiment.

## Upstream validation and remaining counts

The local suite has **193 passing tests**. The CuPy selection excluding
complex128 has **495 passed, 2,608 skipped, 36 expected failures and 36
unexpected passes**: all 3,175 recorded outcomes match the corrected baseline,
including the 24 real SM fixes. The unchanged complex paths were not rerun;
the previous seven complex failures remain open.

The initial all-matched Spectra sweep completed 60 cases: **50 passed and 10
failed**, with 56 passing original numerical bounds and no numerical
regressions. It produced nine new strict passes but lost two strict cycle
passes. Its failed regression gate is preserved in the
[initial comparison](../../results/matching_initial_contracts.json) and
[complete initial rows](../../results/matching_spectra.json).

The final policy retains matching on 27 cases: **23 passed, four failed**;
25 meet the original numerical bounds. Those 27 have nine new passes and no
lost numerical or strict passes. All three sparse standard SM cases and all
six sparse generalized SM cases now pass. The two remaining numerical failures
are `SymGEigsShift_5` and `_30`; `_15` and `_45` pass the original numerical
bounds but have short counts. See the [selected cases](../../results/matching_selected_spectra.json)
and [passing selected-case regression gate](../../results/matching_contracts.json).
Those remaining generalized failures are still tracked in #7.

Standard shifted families restore their prior factor settings rather than
adopting a blanket change. A fresh run of the three cycle examples under that
**original profile** also reports 2/3, 4/5 and 5/6 inverse counts, while their
physical counters are 3/3, 5/5 and 6/6. Every physical residual and requested
spectrum check passes. This establishes sensitivity across captured runs even
with the original profile; the initial results do not isolate matching as the
cause. No favorable retry replaces a failed row. The
[restored-profile cycle results](../../results/matching_cycle_legacy.json)
retain all three failures, tracked in
[#36](https://github.com/alecjacobson/warpack/issues/36).

The [composite inventory](../../results/matching_spectra_overlay.json) overlays
those 27 matched cases and three restored cycle cases onto the previous full
inventory, retaining the other 330 rows: **142 passed, 40 failed, 176 unsupported,
two timed out**. This is not a full new run. It has 159 numerical passes, 23
numerical failures and 178 unevaluated cases. **Its strict regression gate still
fails** for `Example1Shift_1` and `_5`, with no lost numerical bounds; see the
[composite comparison](../../results/matching_overlay_contracts.json). The
[validation record](../../results/matching_validation.json) preserves source
hashes, commands, the initial local failure and both comparison scopes.

## Application accuracy and performance

All 18 application rows across nine API classes at the two standard sizes,
and all three 10,000-unknown A/B/C rows, pass every measured accuracy trial.
All 19 eligible comparisons to the pre-correction multiplicity baseline pass
the 1.25-times slowdown gate. H is numerically accepted but was inaccurate in
that older baseline, so its two timings are not eligible speed comparisons.
The changed B/G paths use matched general factors with zero correction;
other application profiles are unchanged.

| Interior B unknowns | Previous corrected profile | Matched profile | Maximum normalized residual |
|---|---:|---:|---:|
| 80 | 18.205 ms | 12.619 ms | 4.20e-15 |
| 1,024 | 24.183 ms | 17.050 ms | 3.52e-13 |
| 10,000 | 31.531 ms | 21.136 ms | 3.07e-11 |

Each timing is the median of three uncontended replays, with the same matrix,
`k=4`, `ncv=32`, five outer cycles and independent `1e-8` acceptance gates.
Setup and capture are separate. At n=10,000 the residual is above the stricter
internal `1e-11` tolerance, so the reported count is 0/4; it is not relabeled
as full internal convergence. It passes the same external accuracy gate used
by every benchmark backend.

[Standard-size results](../../results/matching_applications.json) and
[10,000-unknown results](../../results/matching_large.json) include all samples,
residual/spectrum/orthogonality checks, factor settings and graph audits.
The interior overhead tracked by [#34](https://github.com/alecjacobson/warpack/issues/34)
is recovered against the pre-correction baseline, not merely improved relative
to the slower corrected configuration. The SPD Anvil and FSAI-CG profiles are
unchanged and were not rebenchmarked for this change.

## Reproduce

Use the [pinned environment and exported Spectra fixtures](README.md).
Run GPU jobs sequentially. Factor profiles are explicit; setup and capture are
outside replay timing, and the application harness validates every trial.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest tests -q
python -m benchmarks.upstream.inverse_profiles --mtype general --matching \
  --steps 0 1 --out /tmp/matching-profiles.json
python -m benchmarks.upstream.inverse_profiles --mtype general \
  --reordering colamd --steps 0 --out /tmp/global-pivoting-profiles.json
python -m benchmarks.applications.compare --methods warp \
  --out /tmp/matching-applications.json --baseline results/multiplicity_applications.json
python -m benchmarks.applications.compare --methods warp --classes ABC \
  --sizes 10000 --steps 5 --out /tmp/matching-large.json \
  --baseline results/multiplicity_large_regression.json
WARPACK_UPSTREAM_REPORT=/tmp/matching-cupy-real.json python -m pytest -q \
  -p benchmarks.upstream.cupy_plugin build/test_upstream_cupy_linalg.py -k 'not complex128'
```

The initial sweep covered 60 real inverse cases. Reproduce the final 27-case
matched-profile subset using the last two selections below; the first command
checks the 33 standard shifted cases under their restored profile. Use a new
output file and compare against the corresponding recorded baseline:

```bash
python -m benchmarks.upstream.bounded spectra --timeout 150 \
  --families SymEigsShift,Example1Shift --out /tmp/restored-shifted.json
python -m benchmarks.upstream.bounded spectra --timeout 150 --selection 4 \
  --families SymEigs,SymGEigsCholesky,SymGEigsRegInv --out /tmp/matching-spectra.json
python -m benchmarks.upstream.bounded spectra --timeout 150 --selection 0 \
  --families SymGEigsShift --out /tmp/matching-spectra.json
python -m benchmarks.upstream.compare_reports \
  --baseline results/matching_selected_baseline.json \
  --current /tmp/matching-spectra.json --out /tmp/matching-contracts.json
```
