# First-metric certificate investigation (#171)

This is a component experiment, not a changed library path. The production
first whitening still reveals rank with canonical eigendecomposition. All
experimental numerical kernels are Warp; NumPy/SciPy are independent checks
outside the solver. Cholesky success alone never certifies rank.

## Certificate and boundaries

For the normalized, symmetric FP64 metric G, form a triangular transform R
from an attempted Cholesky factor, then compute T = fl(fl(R G) Rᵀ). Reject
nonfinite/asymmetric inputs, failed factors, invalid transforms and uncertified
bounds. R must be finite triangular with positive diagonal. The intended bound is
δ ≥ ‖R G Rᵀ − I‖∞. Since the exact congruence is symmetric, its spectral error
is bounded by δ. Consequently

```
lambda_min(G) >= (1-delta) / (norm_1(R) * norm_inf(R))
lambda_max(G) <= norm_inf(G).
```

Accept only when δ ≤ 0.25 and the lower bound strictly exceeds the existing
relative rank tolerance times the upper bound. This preserves every direction;
it cannot replace canonical rank revelation on a deficient or uncertain metric.
The original G and outer solver status remain untouched by the component.

The implementation inflates row/column norms and the measured error. Its
matrix-product error allowance is γ(2+γ)‖R‖∞‖G‖∞‖R‖₁, using a conservative
accumulation-depth budget above the padded 256-entry FP64 tile products and
atomic chunk sum. It separately covers diagonal subtraction, row sums and final
scalar arithmetic. Underflow allowance assumes normalized search metrics; a
future integration must enforce that input scale as well as the certificate.
This is experimental numerical analysis plus testing, not a formal verification
of generated device instructions. No shifts or tolerance changes are allowed.

The component currently owns five dense scratch matrices. Storage sharing,
canonical fallback dispatch and whole-solve trajectory effects remain to be
qualified before any core integration.

## Initial regression coverage

**47 tests pass on pinned Warp 1.18 and stock Warp 1.15**, with logs in
[validation](validation/). Tests include sizes 8–512, strided input/sentinels,
changed and failed replays, indefinite/nonfinite/asymmetric matrices, randomly
rotated spectra across the rank threshold, invalid thresholds, and an injected
inaccurate factor falsely reporting success. Independent SciPy eigenbounds and
extended-precision congruence checks validate every accepted test direction.
Graph DOT inspection finds no host nodes or non-device copies.

## Actual first-Gram observations

The sampler only observes the normalized Gram before the existing canonical
solve. It does not use its certificate to choose any eigensolver operation.
All eight complete first/later outputs pass their native and physical gates.
The table describes the last replay per worker, including its initial zero
search metric. All saved certificate bounds pass independent spectral checks.

| Problem | Sampled metrics | Certified full rank | Samples of size ≥96 | Certified among those |
|---|---:|---:|---:|---:|
|small procedural beam|17|5|0|0|
|100-mode beam, compact history|55|23|18|6|
|200-mode beam, compact history|59|19|24|10|
|Anvil, retained history|17|10|0|0|

Every rejected wide sample has at least one exactly zero diagonal from padded
search directions. A cheap diagonal guard can therefore avoid the expensive
attempt on those samples. This is an observed opportunity, not an assumption
that all future failures have that form.

The instrumented solve reports explicitly set `timing_eligible=False`: extra
sampling/certification work makes those durations unsuitable for solver speed
claims. Generated Gram snapshots are stored externally with their hashes and
reproducible sampling commands in [samples](samples/); no imported mesh is added.

## Isolated cost screening

On the saved Grams, each component restores its input inside the graph. The
canonical path includes its rank decision and whitening coefficients; the
certificate includes factorization, triangular inverse, both products and the
bound. Three batches of three replays per snapshot, L40, pinned runtime,
four Xeon 8362 cores, 2026-10-23; all occupancy records are clean. Original
equations/full spectrum/orthogonality/rank and extended-precision congruences
pass after warmup and every measured batch.

| Snapshot sequence | Sum of canonical medians s | Certificate + canonical on rejection s |
|---|---:|---:|
|100-mode metrics ≥96|0.1824|0.1295|
|200-mode metrics ≥96|0.6233|0.4543|
|Anvil metrics ≥32|0.0272|0.0112|

These sums screen potential savings on a **fixed canonical trajectory**. They
exclude physical basis rotation, conditional dispatch, setup/capture and changes
to outer convergence. They are not complete eigensolve timings. The unimplemented
zero-diagonal guard is not credited. Full-rank 408-dimensional samples take
14.65 ms for certification versus 36.90–47.66 ms for canonical whitening; most
of the complete 200-mode solve remains elsewhere.

Raw samples, checks, exact source/dependency identities and graph audits are in
[components](components/). Reproduce with `sample_first_metrics` followed by
`profile_first_metric` under `benchmarks.experimental`; use the exact commands
in each occupancy JSON. The next step is a guarded experimental composition
with unchanged canonical fallback and complete first/later solve comparisons.
