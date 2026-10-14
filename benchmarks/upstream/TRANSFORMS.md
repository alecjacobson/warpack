# Generalized restarted Krylov transforms

[Matched cuDSS factors](MATCHING.md) fixed sparse inverse accuracy, but block
subspace iteration still missed two required generalized Spectra residual
bounds at 100 cycles. Restarted Krylov now covers standard generalized
shift-invert, buckling and Cayley selection in the appropriate SPD metric.
This resolves the remaining cases in [#7](https://github.com/alecjacobson/warpack/issues/7)
without relaxing assertions or requested subspace sizes.

`GeneralizedKrylovSchur` already supplied the ordinary mass-orthogonal Krylov
path; the generalized upstream adapter now selects it. The production additions
are `BucklingEigensolver(..., method="krylov")` and
`GeneralizedKrylovSchur(..., mode="cayley", target=sigma)`. Both require a
one-RHS inverse of `A - sigma*B`. Legacy block APIs and defaults remain available.
All iteration arithmetic is Warp, with cuDSS used for the sparse inverse.

| Mode | Iteration operator / equivalent pencil | SPD metric | Physical selection |
|---|---|---|---|
| Shift-invert | `(A-sigma*B)^-1 B` | B | Nearest sigma |
| Buckling | Reciprocal pencil `B x = mu A x`, target `1/sigma`; inverse `-sigma*(A-sigma*B)^-1` | A | Largest `abs(lambda/(lambda-sigma))` |
| Cayley | `(A-sigma*B)^-1 (A+sigma*B)` | B | Largest `abs((lambda+sigma)/(lambda-sigma))` |

Buckling and Cayley require nonzero sigma. Buckling permits indefinite B but
requires SPD A. Returned values, vectors and residuals refer to the original
pencil. The transformed selection is not replaced by nearest physical distance.
Permanent tests include positive/negative shifts and a Cayley example where
nearest-shift selection would return the wrong eigenvalue.

## Accuracy and regressions

All **203 local tests pass**, including ten new tests. Exact sparse generalized
fixtures at n=10/100/1000 check physical residuals, requested spectra, metric
orthogonality and invariant subspaces. Exact shifted and buckling/Cayley
regressions use the upstream k/ncv and check full counts over repeated graph replays.
They use 20 fixed cycles. The upstream rerun retains its original 100 cycles.

All **21 supported generalized Spectra cases pass**, including all four
remaining failures. No previous numerical or strict pass is lost in this
subset. The original generic upstream assertions check residuals and counts;
requested-spectrum/subspace checks come from the additional permanent tests
and independent benchmark references, not an invented upstream assertion.

| Previously failing case | New maximum absolute residual | Reported count |
|---|---:|---:|
| SymGEigsShift_5, shift-invert | 2.04e-14 | 10/10 |
| SymGEigsShift_15, shift-invert | 1.93e-14 | 10/10 |
| SymGEigsShift_30, buckling | 3.06e-14 | 10/10 |
| SymGEigsShift_45, Cayley | 1.17e-14 | 10/10 |

[Upstream rows](../../results/transformed_spectra.json),
[contract comparison](../../results/transformed_contracts.json), and
[source/validation record](../../results/transformed_validation.json) preserve
budgets and evidence. A [composite inventory](../../results/transformed_spectra_overlay.json)
overlays these 21 rows onto the last report, retaining 339 other rows:
**146 passed, 36 failed, 176 unsupported, two timed out**. It is not a full new
run. Existing standard shifted/general/complex failures, including #35/#36,
remain visible. CuPy paths and Anvil/CG were unchanged and were not rerun.
The 10k A/B/C performance regression check passes all three rows, with ratios
1.0007–1.0017 against the matched-factor baseline and every trial accurate.

## Matched-accuracy comparisons and fixed budgets

Three synchronized, uncontended trials per row are checked independently in
the original equations, including absolute residual below 1e-9, spectrum,
and metric orthogonality. Timings exclude setup, capture and output conversion.
Native ARPACK/Spectra request tolerance 1e-11; Warp requests 1e-12. All use the
recorded k/ncv, but their starting vectors and stopping mechanisms differ.
Both Warp methods use matched general cuDSS factors with one fixed correction.
CuPy's adapter does not provide these generalized transformed eigsh modes.

| Fixture | ARPACK | Spectra | Warp block, 100 cycles | Warp Krylov, accepted tuned budget |
|---|---:|---:|---:|---:|
| Shift_5, n=100, k=10 | 5.496 ms | 6.166 ms | Fails accuracy | 19.991 ms / 10 cycles |
| Buckling_30, n=100, k=10 | 4.634 ms | 5.554 ms | Fails accuracy | 20.043 ms / 10 cycles |
| Cayley_45, n=100, k=10 | 5.879 ms | 6.184 ms | 46.117 ms | 20.556 ms / 10 cycles |
| Coupled buckling, n=10,000, k=4 | 23.323 ms | 15.418 ms | 145.724 ms | 7.475 ms / 1 cycle |

The n=100 cases use ncv=20; the 10k case uses ncv=32. The CPU references remain
faster on the small cases. Block Cayley passes the external accuracy gate but
reports only 8/10 at its tighter internal tolerance; the accepted Krylov rows
report all requested pairs. No speedup is claimed over failed block timings.
The 10k example is a synthetic sparse congruence-coupled load-factor pencil,
not an additional tetrahedral Anvil benchmark.

The first 20-cycle Krylov runs took 38.55–39.68 ms on the small fixtures and
124.69 ms on the 10k fixture. Reducing a fixed budget is consequential because
captured cuDSS loops execute that budget even after physical convergence.
A controlled sweep found that **five cycles fail all three small fixtures**;
ten pass. The 10k fixture passes at 1/2/3/5/10/20 cycles, with full counts and
all trial residuals below 1e-12. The one-cycle CLI recheck gives the table's
7.475 ms median (about 2.06× faster than Spectra, 3.12× than ARPACK).
These are explicit fixture-specific settings, not a universal one-cycle default
or a new host convergence decision. Revalidate when the matrix/start changes.

[Initial comparisons](../../results/transformed_comparisons.json),
[5/10-cycle sweep including failures](../../results/transformed_budget_sweep.json),
[1/2/3-cycle large sweep](../../results/transformed_large_budget.json), and
[public-CLI large recheck](../../results/transformed_large_cli.json) retain all
samples. Graph DOT audits show no host callbacks or DtoH nodes, but still
contain vendor cuDSS HtoD copies (#29). They are not CUDA API synchronization traces.

## Reproduce

Use the [pinned environment and Spectra fixtures](README.md), and run GPU jobs
sequentially. Native comparisons require the compiler/Eigen/Spectra setup there.

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m pytest tests -q
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 \
  -Ibuild/upstream-spectra/include benchmarks/applications/spectra.cpp \
  -o build/application_spectra.so
python -m benchmarks.upstream.bounded spectra --timeout 150 --selection 4 \
  --families SymGEigsCholesky,SymGEigsRegInv --out /tmp/transformed-spectra.json
python -m benchmarks.upstream.bounded spectra --timeout 150 --selection 0 \
  --families SymGEigsShift --out /tmp/transformed-spectra.json
python -m benchmarks.upstream.compare_reports \
  --baseline results/transformed_baseline.json --current /tmp/transformed-spectra.json \
  --out /tmp/transformed-contracts.json
python -m benchmarks.applications.transformed_regression --out /tmp/transformed-comparisons.json
python -m benchmarks.applications.transformed_regression --methods warp-krylov \
  --krylov-cycles 5 --out /tmp/five-cycle-diagnostics.json
python -m benchmarks.applications.transformed_regression --methods warp-krylov \
  --krylov-cycles 10 --out /tmp/ten-cycle-diagnostics.json
python -m benchmarks.applications.transformed_regression --problems coupled_buckling \
  --methods warp-krylov --krylov-cycles 1 --out /tmp/large-buckling.json
```
