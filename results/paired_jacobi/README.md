# Paired Jacobi output ownership (#172)

This pure-Warp experiment gives one thread each 2×2 output block, reusing the
four matrix entries and right-rotation intermediates. The separate angle kernel,
round/pair schedule, each output's FP64 expression, ping-pong buffers, symmetry
restoration, 30-sweep budget and 2e-15 stopping criterion remain unchanged.
No persistent single-block execution or reordered block-Jacobi schedule is used.

For an even dimension, each round partitions all indices into disjoint pairs.
The Cartesian product of those pairs covers every output entry exactly once.
Input/output buffers remain distinct, so no writer races with another reader.
Round tests verify bit-identical H/V outputs, including strided storage and
sentinels, at block sizes 64/128/256. Full solves also match the existing solver
bit for bit through dimension 612 on scaled indefinite and repeated/null spectra.

## Isolated controls

L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23. Three measured replays
plus validated first replay per dimension/backend, including device input reset.
Setup/capture and host reference checks are separate. All three occupancy
records are clean, with matching inputs and source/dependency identities.

| Dimension | Existing global ms | Paired64 ms | Paired128 ms | Paired256 ms |
|---|---:|---:|---:|---:|
|48|1.654|1.671|1.733|1.959|
|104|4.170|4.245|4.328|4.889|
|208|11.562|12.207|11.380|12.254|
|256|15.452|14.436|14.722|15.608|
|312|22.083|20.143|20.836|21.078|
|360|28.688|25.940|26.196|26.531|
|408|36.256|32.837|33.550|35.182|
|512|60.197|53.282|54.679|55.065|
|612|81.632|75.016|74.793|77.536|
|1024|248.187|206.139|211.696|213.005|

These indefinite cases favor 64 threads from dimension 256, with 7–17% component
improvement; smaller sizes generally lose. Repeated/null controls through 612
show the same broad tradeoff. All 240 first/measured component outputs pass
original equations, complete spectra, full-basis orthogonality and native stopping.
[Initial controls](isolated/) and [midrange/1024 controls](midrange/) retain every
sample and graph audit. Their repeated/null fixture differs from the expanded
1024 regression described below; no failed case is being counted as accepted.

## Whole-solve controls

Only existing even global projections of size 256–1024 switch to paired64.
Other paths, including Anvil's small projections, remain unchanged.

| Problem | Capture s, default / paired | First-compute s, default / paired | Warm median s, default / paired |
|---|---:|---:|---:|
|100 modes, compact history|1.3849 / 1.4162|4.5855 / 4.6076|2.6691 / 2.6621|
|200 modes, compact history|3.8705 / 3.8819|13.7797 / 13.6473|8.8999 / 8.6928|
|Anvil, retained history|0.6071 / 0.6045|3.0211 / 3.0863|1.6168 / 1.6269|

All 24 complete first/later outputs pass native and physical/requested-space
checks. Every pair has matching inputs, starts, source/version/dependencies and
clean occupancy. The 200-mode replay gain is 2.3%; the 100-mode difference is
small. Anvil executes the same projected path and varies slightly in stopping
iteration. No Anvil speedup is claimed. Separate setup phases are in the raw
[whole controls](whole/); first-compute includes setup, capture and the first
validated captured solve, excluding assembly/reference/imports/host checks with
cached kernels. These results do not replace the six-backend showcase table.

## Known expanded regression and next step

35 initial/expanded tests pass on pinned Warp; the first 33 also pass on stock.
Additional 1024 testing passes the indefinite case but exposes an existing
repeated/null full-basis Frobenius orthogonality error of **1.18009e-11** against
the unchanged 1e-11 gate. Default and paired results are bit-identical. This is
tracked separately in [#174](https://github.com/alecjacobson/warpack/issues/174),
with the [original failing log](validation/1024-known-failure.log). The regression
is marked strict xfail pending that fix, not claimed as a passing component.
Column normalization is under investigation; it must not hide off-diagonal loss
or loosen the gate. No production dispatch changes have been made here.

Reproduce with `benchmarks.experimental.paired_jacobi_compare` and
`compare_paired_jacobi`, using the exact commands in occupancy JSONs. The
whole-solve gain remains much smaller than the isolated projected-solver gain.
