# Compact general-spectrum discovery experiments

Issue #104 remains open. These fixed-budget experiments do not repair the
original compact requested-spectrum failures. No production algorithm or
acceptance threshold changes, and no timing claim is made.

The public checkpoint policy preserves a completed set; it cannot certify that
the iteration discovered every requested mode. A fresh original-width audit
with checkpoints and **300 fixed cycles** passes 12/20 cases on all three
replays: all ten n=100 cases and large cases 32/35. The original compact
adaptive profile passes 11/20. The previously reported wider ncv=40/60,
fixed-80 profile passes all 20; it is a different configuration.

[Complete fixed-300 audit](../../results/performance_followup/public-checkpoint-compact-fixed300.json)
retains every accepted and rejected trial. Case 30 passes only its first
replay; a profile is accepted only when every replay passes. Cases 31, 34,
36–39 return complete, low-residual sets with missing requested modes. Case 33
has a short count. Extra iterations alone are insufficient.

## Diagnostic policies

Every row below uses the original matrix/start, k=10, ncv=30, LM, tol=1e-10,
300 fixed cycles, three reset replays, and the same independent reference.
The reference requires injective assignment, every mode strictly above the
boundary, original residuals, unit norms, independent vectors and the requested
invariant subspace. Experiments preserve the public completed-set snapshot.

| Policy | Cases tested | Cases passing all replays |
|---|---|---|
| Continue with the largest unfinished retained-guard residual after requested convergence | 30, 31 | none |
| Insert a new independent direction each cycle after requested convergence | 30, 31 | none |
| Retain and advance the best-ranked unfinished extra Ritz vector | 30, 31 | none |
| Restart the whole working basis independently every 100 cycles | 30, 31, 33 | 33 |
| Orthogonal iteration retaining the entire ncv-dimensional basis | 30, 31, 33 | none |

The first three policies still return full-count but incorrect spectra on both
cases. Fresh single directions or one protected unfinished vector do not solve
the demonstrated discovery problem. This is evidence against these particular
policies, not a proof that exploration cannot help.

Independent rounds improve case 33 to full acceptance. Case 30 is a useful
negative control: residuals are at most 1.73e-11 and injective assignment error
is below 2.57e-11, yet it omits the mandatory eigenvalue approximately
`-16.46578734 + 7.92360176i` and includes **both** boundary eigenvalues near
`17.67178318 ± 4.56668159i`. The boundary allows either member, but not omission
of a strictly larger-magnitude mode. The mandatory-coverage check correctly
rejects it. Case 31 still returns the wrong set.

Whole-subspace orthogonal iteration does not reach the residual tolerance:
worst normalized errors are approximately .0058, .0244 and .0076. Its cost per
cycle differs from restarted Arnoldi, so this equal-cycle diagnostic is not a
performance comparison. It supports LM only and is not a production fallback.

## Evidence and reproduction

- [Guard residuals](../../results/performance_followup/general-guard-compact30-31.json)
- [Independent directions](../../results/performance_followup/general-independent-compact30-31.json)
- [Unfinished Ritz vector](../../results/performance_followup/general-frontier-compact30-31.json)
- [Independent rounds](../../results/performance_followup/general-rounds-compact30-31-33.json)
- [Whole-subspace iteration](../../results/performance_followup/general-subspace-compact30-31-33.json)

The benchmark-only implementations live beside this report. The profile driver
records solver identity, source hashes, runtime hashes, original data hashes,
width, budget, all physical checks and device counters. Later additions to the
driver add policy flags; each earlier report retains the hash actually tested.
All mathematical kernels remain Warp operations. The restart-round schedule is
fixed when constructing the graph; it does not read convergence on the host.

```bash
python -m benchmarks.experimental.general_selection_profiles \
  --fixtures build/spectra-benchmark-fixtures \
  --guard-search --case benchmark_gen_30 --case benchmark_gen_31 \
  --ncv-multiplier 1 --fixed --budget 300 --repeats 3 \
  --out build/general-search.json
```

Replace `--guard-search` with `--independent-search`, `--frontier-search`,
`--restart-rounds 100`, or `--subspace-iteration`. For the latter two, also add
`--case benchmark_gen_33`. Omit policy flags and case filters to reproduce the
public fixed-300 inventory. Use the pinned Warp environment from the README;
GPU jobs run sequentially. No changes to installed CuPy or CPU numerical
iteration are involved.

Ten focused tests cover requested-mode priority, frontier ordering, changed
matrix replay, reset state, completed-set preservation and no HOST/HtoD/DtoH
graph nodes. They verify implementation contracts on controls; they do not
turn the rejected upstream fixtures into passes.
