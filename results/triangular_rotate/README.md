# Retaining both X-orthogonalization passes (#173)

The first experiment computes the small inverse triangular factor once per
pass, then applies it to X and cached AX with the existing dense tiled rotation.
Both full CholeskyQR passes, initial scaling, image transformations and final
mass application remain. It reuses existing gx/bx scratch; no dense buffers are
added. Production dispatch is unchanged.

**This generic dense-rotation version loses:** about 50% slower for the isolated
stage, so it is not promoted or sent to expensive whole-solve comparisons.
The inverse is triangular; generic dense rotation does unnecessary arithmetic
on its zero upper triangle. Exploiting that structure is a separate possible
follow-up, not a result established by these measurements.

L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23; n=34,560, synthetic
coupled tridiagonal SPD mass. Default block-product settings, not the tuned FEM
recipe. Each captured stage restores the same device X/AX inputs and performs
the complete two-pass orthogonalization. Five warm replays after a separately
checked first replay, cached kernels, fixed baseline-then-candidate order;
continuous occupancy clean.

| Width | Input | Sequential triangular stage ms | Inverse + dense rotation stage ms |
|---|---|---:|---:|
|104|Random basis|6.5389|9.8340|
|104|Already mass orthonormal|6.5373|9.8451|
|204|Random basis|26.1577|39.2410|
|204|Already mass orthonormal|26.1744|39.2313|

All **48 first/measured stage outputs pass** status, fresh M-orthogonality and
independent AX/MX checks. Maximum Frobenius orthogonality error is below 6.1e-15.
Graph audits show no host nodes or non-device-to-device transfers. This is a
component result, not an eigensolve or FEM speed claim; allocation, compilation,
capture and host validation are outside stage timing.

Eight pinned tests additionally cover widths 8/52/104/204, with and without
cached images, changed mass, physical row scaling by 1e±60, correlated inputs,
and a mass matrix equilibrated with diagonal factors spanning 1e±6. Both
methods retain the complete physical subspace and cached-image consistency.
[Transcript](validation/tests.log); adjacent reports retain exact source/input
identities, commands, samples and occupancy. The candidate remains outside core.

## Existing triangular kernel: thread grouping

A second screen changes only the original kernel's block size. Twelve tests
verify bit-identical triangular outputs at widths24/104/204 for32/64/128/512
threads versus the256-thread baseline, including strided storage, scale1e±80,
zero/changed replay and independent triangular equations.

All120 first/measured full-stage outputs pass original M-orthogonality/AX/MX
checks; occupancy is clean. Same fixture/protocol as above. Warm stage medians
in milliseconds:

| Width/input | 256 threads | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
|104/random|6.535|6.443|6.511|6.496|7.006|
|104/orthonormal|6.538|6.443|6.517|6.499|7.011|
|204/random|26.074|26.031|26.075|26.103|26.991|
|204/orthonormal|26.106|26.018|26.094|26.123|27.023|

The widest problem shows no substantive gain. The small104-width improvement
is about0.09ms for this stage, not a demonstrated whole-solve improvement.
512threads loses. This does not justify a production policy or expensive
whole-FEM tuning runs. [Raw controls](block-size-a1.json) and
[exact-output tests](validation/block-size-tests.log) retain the experiment.

## Skipping the inverse's zero upper triangle

The next variant skips coefficient tiles strictly above the inverse's lower
triangle. It preserves both orthogonalization passes and reuses the same scratch.
Twenty-one tests pass, including odd/partial tiles, strides/sentinels, exact
agreement with the dense product, coupled/scaled/changed mass and cached images.
All72 first/measured stage outputs pass; continuous occupancy is clean.

| Width/input | Original triangular ms | Dense inverse application ms | Lower-triangular inverse application ms |
|---|---:|---:|---:|
|104/random|6.539|9.838|7.614|
|104/orthonormal|6.534|9.837|7.617|
|204/random|26.095|39.717|29.905|
|204/orthonormal|26.064|39.733|29.907|

Avoiding zero work helps but still leaves the full stage roughly15–17% slower
than the original solve. This version is also **not promoted**; no whole-FEM
speedup is claimed or assumed. [Raw controls](lower-a1.json) and
[test transcript](validation/lower-tests.log) preserve the result. The initial
generic variant now calls a small overrideable application method; its algorithm
is unchanged, and the original committed source is retained in32af963.

## Direct blocked substitution

A fourth candidate forms **no inverse**. Each CUDA block owns 32 physical columns;
8-row panels use tiled products against completed panels and forward substitution
inside the diagonal panel. Both QR passes and all other stage work remain.
No new dense workspace is allocated. The summation order changes, so equality
with the serial implementation is judged by the original triangular equations
and physical subspace, not bitwise identity.

Fourteen pinned tests pass ([transcript](validation/blocked-tests.log)), also
passing on stock Warp. Coverage includes odd/partial tiles, strided storage and
sentinels, changed factors and replay, scales 1e±80, both cached-image policies,
coupled/changed mass, correlated inputs and equilibrated mass with scales 1e±6.
All **48 first/measured full-stage outputs pass**, with clean continuous GPU
occupancy, matched current source/input/dependency identities and device-only
graphs. Same component protocol as above:

| Width/input | Original triangular ms | Direct blocked triangular ms |
|---|---:|---:|
|104/random|6.541|5.922|
|104/orthonormal|6.545|5.910|
|204/random|26.171|23.171|
|204/orthonormal|26.186|23.176|

This is roughly 9–11.5% faster for this isolated stage. [Raw controls](blocked-a1.json)
retain every output, sample and source identity. The whole-solve controls below
measure the corresponding total effect. The candidate remains outside core.

### Complete solves with the 8-row panels

All **24 complete first/later outputs pass** for 100 modes, 200 modes, and 200
modes with compact projected graphs. Matched current source/input/dependency
and independently verified CPU reference identities, clean occupancy, unchanged
native stopping and original spectrum/subspace/rigid-mode gates. Both methods
use the installed first-whitening certificate. Each worker constructs its own
preconditioner and solver; the reference-only cache is outside backend timers.
The triangular candidate is enabled only for block widths 96–256.

| Problem | Application | Setup s | Capture s | First-use compute s | Warm median s |
|---|---|---:|---:|---:|---:|
|100|Original triangular|0.4409|1.4382|4.5088|2.5132|
|100|Direct blocked|0.4715|1.5769|4.6672|2.4659|
|200|Original triangular|0.6304|3.9207|13.3907|8.4010|
|200|Direct blocked|0.6380|4.0451|13.3750|8.2219|
|200-compact|Original triangular|0.6501|1.7013|10.8333|8.4780|
|200-compact|Direct blocked|0.6486|1.7535|10.7931|8.2553|

First-use compute includes setup, capture/eager preparation and the first
captured solve; it excludes matrix assembly, CPU reference preparation, imports
and host validation. Warm medians use three later replays, with every output
checked. Setup/capture are individual observations with cached compilation where
available. This is not process wall time.

Warm improvements are about 1.9%, 2.1%, and 2.6%, respectively. Iteration counts
are 54/55/54→54/54/54, 58/57/58→57/58/57, and 58/58/58→57/57/57;
the observed gain includes floating-point trajectory changes. First-use time
does not improve consistently. [All six raw controls](whole/) include commands,
source identities, full quality measurements and occupancy. The broader FEM-family and panel-size controls below determine whether this
gain carries across work shapes.

### Broader controls and the remaining regression

All 24 additional free/disconnected/heterogeneous first/later outputs pass
unchanged native/original requested-space checks, matched source/input/dependency/
reference identities and clean occupancy. Medians:

| Family | Original triangular s | Direct blocked s |
|---|---:|---:|
|free|2.39867|2.36478|
|disconnected|4.41630|4.45410|
|heterogeneous|2.75561|2.71112|

Free and heterogeneous improve about 1.4%/1.6%, but disconnected is **0.9% slower**.
Both disconnected variants take 32 outer steps on every measured replay. The
[reverse-order recheck](recheck/) also passes all eight outputs, with clean
occupancy and matched identities, and confirms the slowdown:
original 4.40025s versus blocked4.43593s. The first rejected performance result
is retained; the repeat does not replace it.

A separate [component control at the disconnected shape](blocked120-large-a1.json),
70,656 rows and width 120, passes all24 first/measured outputs but confirms a
stage regression: 14.736→15.801ms for random inputs and 14.789→15.823ms for already
mass-orthonormal inputs. This supports a work-shape-dependent cost, rather than
attributing the whole result solely to convergence variability. **The direct
blocked implementation is not promoted to the library.** Further blocking or
principled dispatch qualification is needed; correctness alone is insufficient.

### Larger row panels

Two further component screens retain both full QR passes and use 16 or 32 row
panels. All 96 first/measured outputs pass with clean occupancy; these remain
benchmark-only variants. Each row has its own fresh serial control:

| Row panel | Width/input | Original stage ms | Blocked stage ms |
|---|---|---:|---:|
|16|104/random|6.541|6.223|
|16|104/near_orthogonal|6.540|6.156|
|16|204/random|26.187|22.780|
|16|204/near_orthogonal|26.210|22.795|
|32|104/random|6.533|7.447|
|32|104/near_orthogonal|6.533|7.503|
|32|204/random|26.107|25.595|
|32|204/near_orthogonal|26.118|25.612|

The 16-row panel slightly helps the 204-width stage but loses to 8 rows at 104;
32 rows are worse at both sizes. No whole-solve gain is claimed for these
variants. [16-row](panel16-a1.json) and [32-row](panel32-a1.json) controls preserve
samples, source and input identities, commands and occupancy.

Expanded validation: **47 pinned passes**, **37 stock passes/10 expected FSAI
feature skips** ([pinned](validation/combined-tests.log),
[stock](validation/combined-stock-tests.log)). These include the panel variants,
unread NaN-poisoned upper factor triangles, complete small FEM families with
both history policies, and changed coupled mass/constraints/clustered modes.
The original triangular path, both whitening passes and all stopping thresholds
remain unchanged in the installed library.

### Larger physical-column tiles and the copy-out race (#178)

The first 64-column large-shape trial **failed accuracy** on every candidate
output: M-orthogonality errors 0.0034–0.0061 and cached-image errors 0.0096–0.0127,
despite status 0. The 12 serial controls passed. Its [rejected report](columns64-large-a1.json)
exits nonzero and is not timing evidence. Earlier 42-case pinned/stock tests
covered strided/odd storage but missed this larger dense, even-stride path.

The post-store scratch write could race with Warp's vectorized shared-to-global
copy: one thread may copy two doubles owned by different lanes. The scatter
writes before its trailing barrier. Replace that scratch mutation with a
masked-off scatter, which supplies a barrier without overwriting any entry.
The [archived failing source](validation/columns-unfixed-source.txt) matches the
original report hash. A [reduced identity-solve test](validation/copyout-reproducer.log)
reproduces 512 corrupt entries out of 8,478,720 at 70,656 columns. The installed
library's serial triangular kernel was never changed by this experiment.

After the fix, **46 tests pass on both pinned and stock Warp**, including repeated
changed identity solves at dense even shapes with 64/128 threads, original
partial/strided shapes, unused NaN upper factor entries, coupled/scaled mass and
both full QR passes. [Pinned checks](validation/columns-fixed-tests.log),
[stock checks](validation/columns-fixed-stock-tests.log).

All **192 fresh first/measured component outputs pass** after the fix, with
clean occupancy and matched source/input/dependency identities. The timings
below are exclusively the corrected implementation:

| Physical columns | Rows | Width/input | Original stage ms | Corrected blocked stage ms |
|---|---:|---|---:|---:|
|64|34560|104/random|6.546|6.944|
|64|34560|104/near_orthogonal|6.543|7.007|
|64|34560|120/random|8.678|9.196|
|64|34560|120/near_orthogonal|8.680|9.191|
|64|34560|204/random|26.177|27.161|
|64|34560|204/near_orthogonal|26.205|27.072|
|64|70656|120/random|14.769|18.198|
|64|70656|120/near_orthogonal|14.726|18.178|
|128|34560|104/random|6.530|6.089|
|128|34560|104/near_orthogonal|6.547|6.020|
|128|34560|120/random|8.684|7.947|
|128|34560|120/near_orthogonal|8.682|7.961|
|128|34560|204/random|26.198|23.484|
|128|34560|204/near_orthogonal|26.192|23.374|
|128|70656|120/random|14.764|15.887|
|128|70656|120/near_orthogonal|14.780|15.886|

64-column tiles are slower throughout. 128-column tiles help the smaller matrix
but still lose on 70,656 rows at width 120. Neither is promoted or sent to
whole-solve runs. Fixed raw controls:
[64/small](columns64-fixed-a1.json), [64/large](columns64-large-fixed-a1.json),
[128/small](columns128-fixed-a1.json), [128/large](columns128-large-fixed-a1.json).
Original reports without `fixed` remain archived; their implementation had the
identified race, even where that particular run passed its physical checks.
