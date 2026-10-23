# Installed first-whitening certificate (#171)

**Qualified:** all 56 complete first/later control outputs pass, with matched
source/input/dependency/reference identities and clean continuous GPU occupancy.
Wide-case warm replay improves 3.6–6.2%; the small Anvil path is unchanged.

The installed block solver can now avoid the first search-metric eigendecomposition
when a fresh device certificate proves that every active direction exceeds the
unchanged relative rank threshold. Both projection/whitening passes remain.
The default specialization applies only during capture on Ada (SM 89), for search
metrics of size 96–512. Smaller metrics, other architectures and eager solves
retain canonical whitening. The standard 20-mode Anvil configuration is below
this guard.

## Rank and failure handling

The first pass forms and normalizes the physical Gram exactly as before. It
recognizes only **exact** zero padding: a zero diagonal must have an entirely
zero row and column. A separate auxiliary matrix fills that padding with identity,
while retaining the original Gram for fallback. The physical output keeps only
the original active mask and rank; auxiliary directions never become modes.

Cholesky, inverse-factor formation and two products produce a measured congruence
error. A conservative bound includes product/reduction roundoff, absolute-sum
inflation and an underflow allowance. Accept only if the resulting lower bound
on the auxiliary matrix's smallest eigenvalue exceeds the unchanged relative
rank threshold times an upper bound on its largest eigenvalue. An additional
exact separation check rejects any computed inverse that couples active and
inactive coordinates. See the [derivation and independent prototype checks](../padded_first_metric/README.md).

An uncertain bound, dependent subspace, failed factor, invalid normalization or
pre-existing outer failure routes to canonical rank revelation on the **original**
normalized Gram. A failed attempted factor has private status and cannot contaminate
an otherwise valid outer solve. The [#176 routing regression](../padded_first_metric/validation/status-reproducer.log)
is included in installed tests: failure remains visible, and changed healthy inputs
recover on the same captured graph.

Both whitening passes share the existing three dense scratch arrays across
mutually exclusive search branches. The first pass reuses the transform and
projected-eigenvector buffers, and reuses factor workspace for a product only
after the factor/inverse launches finish. It adds small flags/masks/diagnostics,
**no dense workspace**, and no host numerical decision, replay synchronization, vendor
call or non-device-to-device graph transfer. All numerical work is Warp.

## Comparison protocol

The comparison driver explicitly selects `--first-whitening canonical` or
`--first-whitening default`; canonical disables only the new first-pass dispatcher.
Both use the same installed second-pass certificate, paired Jacobi updates,
CG/preconditioner settings, inputs and physical acceptance gates. `--compact-graphs`
selects the separately supported compact projected graph composition.

Each worker independently assembles the current matrices and constructs its own
preconditioner and solver. The [explicit CPU reference cache](../reference_cache/README.md)
reuses only independently verified reference arrays, outside backend setup,
capture and solve timing. Content, provenance, reference spectrum/subspace and
original equations are rechecked. Reference identities must match within each
control pair; Anvil retains its separately recorded fixture protocol.

Each configuration checks a separately recorded first solve and three warm
replays, including native status/count, original equations, mass orthogonality,
requested spectrum/subspace and retained rigid modes. Setup and capture are
reported separately. Runs use pinned Warp/FSAI, the L40, four pinned Xeon 8362
cores, cached compilation and continuous GPU occupancy monitoring. Contended or
failed measurements are retained and excluded from accepted timing comparisons.
This is a first-pass comparison, not a replacement for the README's historical
six-backend Anvil table.

## Regression and wheel checks

- Latest complete suite: **2,735 passed, 11 skipped** ([transcript](validation/full-tests.log)).
  This collection preceded the additional eight installed bound/odd-size/512-size
  checks and four comparison-driver checks below.
- **119 focused tests pass** on pinned Warp, including those twelve additions
  ([transcript](validation/focused-tests.log)). Rank/mask/subspace checks cover
  full, exactly padded, dependent, zero, badly scaled and changed inputs. Forced
  factor failure matches canonical output bit-for-bit. Negative-mass and prior
  failure routing, same-graph recovery, strided shared scratch/sentinels, both
  history policies, compact graphs, coupled mass/constraints, clusters and rigid
  modes are exercised. Accepted installed certificates are also checked with
  independent extended-precision congruence bounds.
- Stock Warp passes the same first-whitening checks: **86 passed, 33 expected
  skips** for pinned upstream FSAI/batched-CG features. The [combined transcript](validation/stock-tests.log)
  reports 100 passes because it also includes 14 separate blocked-triangular
  experiment checks; those are not part of the first-whitening implementation.
- The isolated wheel contains **40 core modules**, each byte-identical to the
  tested source; no benchmark/application packages are included.
  [Wheel identity](validation/wheel.json) records SHA256
  `f65edbb5fb9aefcf31766daf0c7e6c31a49a5e53cada326e36fed5f575dd2afb`.
- With benchmark/application/SciPy/CuPy/cuDSS imports forbidden, the extracted
  wheel passes [three public replays](validation/wheel-public.json),
  [sixteen projected outputs](validation/wheel-projected.json), and
  [nine wide public generalized replays](validation/wheel-wide-public.json).
  The latter changes stiffness and mass on the same graph, includes constraints
  and repeated modes, and exercises both history policies and compact graphs.
  Every wide replay uses the new fast path. Worst absolute original residual is
  2.36e-9; mass-orthogonality error is below 8.5e-16. Native counts, analytic
  requested spectra/subspaces and graph-residency checks pass.

The prototype numerical kernels were compared structurally during review before
promotion. Core dispatch, branch ownership, buffer lifetimes, failure status,
preloading, eager fallback and benchmark controls were reviewed separately.
The independent prototype remains outside the library for reproducible controls.

## Fresh installed controls

Seconds, L40, 2026-10-23. **First-use compute = setup + capture/eager preparation
+ first captured solve**, excluding matrix assembly, CPU reference preparation,
imports, input fingerprinting and host validation. Warm median uses three later
replays. Setup/capture are single observations with cached compilation where
available, not statistically established setup improvements. Raw report/occupancy
files beside this document retain the exact commands and individual samples.

| Problem | First whitening | Setup s | Capture s | First-use compute s | Warm median s |
|---|---|---:|---:|---:|---:|
|[100](production/100-canonical-a1.json)|canonical|0.4326|1.4308|4.6483|2.6518|
|[100](production/100-default-a1.json)|default|0.4302|1.4286|4.4905|2.5060|
|[200](production/200-canonical-a1.json)|canonical|0.6216|3.9601|13.7430|8.7059|
|[200](production/200-default-a1.json)|default|0.7135|4.1627|13.7617|8.3536|
|[anvil](production/anvil-canonical-a1.json)|canonical|0.7905|0.6203|3.0485|1.6233|
|[anvil](production/anvil-default-a1.json)|default|0.8279|0.6421|3.1015|1.6287|
|[200-compact](production/200-compact-canonical-a1.json)|canonical|0.6473|1.6700|11.1868|8.8330|
|[200-compact](production/200-compact-default-a1.json)|default|0.6575|1.6865|10.8815|8.4593|
|[free](families/free-canonical-a1.json)|canonical|0.4421|1.5653|4.6998|2.5544|
|[free](families/free-default-a1.json)|default|0.4952|1.6185|4.6497|2.3952|
|[disconnected](families/disconnected-canonical-a1.json)|canonical|0.5129|1.9146|7.2091|4.6136|
|[disconnected](families/disconnected-default-a1.json)|default|0.5248|1.9113|7.0486|4.4479|
|[heterogeneous](families/heterogeneous-canonical-a1.json)|canonical|0.4193|1.3941|4.8224|2.8993|
|[heterogeneous](families/heterogeneous-default-a1.json)|default|0.4342|1.4549|4.7543|2.7550|

The 100/200-mode controls improve warm replay by 5.5%/4.0%; the 200-mode compact
composition improves 4.2%. The broader free, disconnected and heterogeneous
cases improve 6.2%, 3.6%, and 5.0%. Original normalized residuals for these wide FEM
cases remain below 1e-11, with requested-space error below 5.8e-11 and the expected
six/twelve rigid modes retained in the free/disconnected cases. The Anvil
configuration does not allocate or execute the new helper; 1.6233→1.6287 seconds
is an unchanged-path control, not an attributed gain or regression.

Default last-replay usage is 17 fast passes, one canonical fallback and one
included early rejection for 100 modes and the broader families; 200-mode cases
record 23/1/1. These counts cover only eligible wide search branches. The canonical
control and Anvil have no first-pass helper. Three measured outer iteration
counts are retained below; differences reflect floating-point trajectories,
not a changed stopping threshold:

| Problem | Canonical counts | Default counts |
|---|---|---|
|100|54 / 55 / 54|54 / 54 / 54|
|200|57 / 57 / 57|57 / 58 / 57|
|anvil|16 / 16 / 17|16 / 17 / 16|
|200-compact|58 / 58 / 57|58 / 58 / 58|
|free|36 / 36 / 36|36 / 36 / 36|
|disconnected|32 / 32 / 32|32 / 32 / 33|
|heterogeneous|51 / 51 / 51|51 / 51 / 51|

First-use time does **not** improve consistently, and the 200-mode ordinary
setup/capture observations are higher for the default. This qualification
establishes warm solve improvements without hiding those costs. It does not
close the remaining wide-FEM gap against matched-preconditioner CuPy (#147).

Reproduce a control with `benchmarks.experimental.compare_cached_first_whitening`,
`--public-api`, an explicit verified `--reference-cache`, and
`--first-whitening canonical` or `default`; use the complete settings in its
adjacent occupancy command. Add `--compact-graphs` for the compact pair. No
reference or backend factor is silently borrowed from another measured solver.
