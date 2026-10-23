# Paired Jacobi with immutable round indices (#172)

The same paired-output reuse now works inside ordinary and compact projected
graphs. The round comes from **immutable `partners[n-1]`**, not the compact-loop
counter that another thread advances. Each thread owns one disjoint2×2 output
block, retaining the original right-then-left FP64 expressions. The angle
kernel, pair order, symmetry restoration, final column normalization,30-sweep
cap and2e-15 native stopping criterion remain unchanged.

Twenty new tests pass on both pinned and stock Warp; six additional nonfinite
and recovery controls pass on pinned Warp. Ordinary/compact complete H,V,sweep
results are bit-identical, including1024-dimensional repeated/null inputs,
scales1e±80, zero/recovery, strided sentinels and2/16-round compact schedules.
Graph audits exclude host nodes and non-device-to-device copies.

## Isolated controls

L40, pinned Warp/FSAI, four Xeon8362 cores,2026-10-23. All192 first/measured
component outputs pass independent original equations, full spectra and
orthogonality, plus native stopping. Continuous occupancy is clean. Medians
of three warmed replays, milliseconds; fixed backend order and cached kernels.

| Dimension | Indefinite global / paired | Indefinite compact / paired compact | Repeated/null global / paired | Repeated/null compact / paired compact |
|---|---:|---:|---:|---:|
|208|11.562 / 11.681|12.206 / 12.291|21.149 / 21.367|22.315 / 22.475|
|256|15.421 / 14.696|16.202 / 15.582|26.723 / 25.503|28.050 / 27.010|
|312|22.639 / 20.431|23.085 / 21.388|43.535 / 39.342|44.382 / 41.174|
|408|36.344 / 33.022|37.868 / 34.306|69.622 / 63.380|72.546 / 65.756|
|612|83.914 / 75.235|83.869 / 77.156|154.668 / 139.003|154.561 / 142.462|
|1024|248.517 / 209.398|253.706 / 211.480|478.039 / 401.637|487.127 / 406.824|

208 still loses slightly. The whole-solve candidate therefore applies only to
Ada89 even projected dimensions256–1024, with64threads/block. Compact controls
use16rounds/loop. No assertion is made about other GPU architectures.

## Complete solves

All32 first/later outputs pass native count/status/stopping and independent
original-pencil, requested-spectrum/subspace and orthogonality gates. Every
pair has matching inputs, initial vectors, reference data, source and dependency
identities. Internal and continuous external occupancy are clean.

| Problem | Setup s, original / paired | Capture s, original / paired | First-compute s, original / paired | Warm median s, original / paired |
|---|---:|---:|---:|---:|
|100 modes, compact history|0.4041 / 0.4014|1.3826 / 1.4021|4.5889 / 4.6028|2.6791 / 2.6490|
|200 modes, compact history|0.6147 / 0.5989|3.8940 / 3.9001|13.8963 / 13.7104|8.8911 / 8.7484|
|200 modes, compact history and compact projected graphs|0.6038 / 0.6167|1.6562 / 1.6894|11.2974 / 11.2126|8.9990 / 8.7980|
|Anvil, retained history|0.7861 / 0.8090|0.6106 / 0.6123|3.0245 / 3.0537|1.6207 / 1.6249|

The measured200-mode median reductions are **1.6% ordinary /2.2% compact**.
Outer counts remain visible in raw reports: ordinary baseline58/57/57 versus
paired58/58/58; compact58/57/58 versus57/57/57. The100-mode difference is small
and includes54/55-step variability. Anvil uses no eligible projected dimensions;
its smaller path is unchanged, with16/17-step variability and no speed claim.
This is a modest improvement, not closure of the wider #147 CuPy gap.

Setup independently constructs the preconditioner and solver. First-compute
includes setup, capture and the validated first captured solve, excluding
assembly/reference/imports/host validation with cached kernels. Setup/capture
are single observations. These narrow controls do not replace the six-backend
Anvil showcase. All phases, commands, samples, source/input identities and
gates remain in adjacent JSON/occupancy reports. Core promotion is a separate
step; these results describe the experimental wrapper against the normalized
core baseline.
