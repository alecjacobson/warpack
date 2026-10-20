# Cooperative QR loop fusion (#14, #110)

The prototype keeps the complete bounded shifted QR loop in a single Warp
block, with the Hessenberg matrix in shared memory and Schur vectors in global
memory. It preserves the prior deflation threshold, exceptional shifts,
iteration budget, right-vector extraction and original-equation final checks.

The first prototype passes 21 projected tests on pinned Warp and all35
component timing samples. Saved50x50 projections improve from about18.0ms to
13.7ms. However, Warp1.15 rejects its 2D vector-tile scalar indexing (#110).
The corrected prototype uses a flat shared alias for scalar accesses, while
retaining row/column views for cooperative updates. Stock1.15 passes21 checks;
the common old/new suite passes41 on pinned Warp in20.04s.
[Initial component records](../../results/fused-schur-components-initial.json),
[occupancy](../../results/fused-schur-components-initial.occupancy.json),
[compatibility failure](../../results/performance_followup/fused-schur-stock-failure.json).

After the correction, both complete #35 fixtures pass all20 requested pairs
on each of three synchronized warm replays, with the original tolerances,
widths and budgets. Direct SR median is2.889955s (80–81 cycles), and cuDSS
shift-invert is1.161794s (fixed30). Earlier accepted tiled medians were3.216071s
and1.287696s. These sequential measurements suggest roughly10% improvement;
they are not alternating paired samples, and unlocked clocks plus direct
iteration variability limit precision. They say nothing about unresolved
LM selection failures in #104 or large sparse general performance.
[Direct](../../results/fused-schur-direct.json),
[direct occupancy](../../results/fused-schur-direct.occupancy.json),
[shift](../../results/fused-schur-shift.json),
[shift occupancy](../../results/fused-schur-shift.occupancy.json).

The direct graph has one outer conditional, no HOST nodes and only DtoD
copies. The fixed shifted graph has no conditionals or HOST nodes; cuDSS's
existing small HtoD metadata copies remain (#29). The prototype's constructor
still prepares an unused old QR graph. Public integration must remove this
setup waste, retain a shared-memory capacity fallback, and rerun appropriate
suite/packaging checks before promotion. Checkpoint selection is separate.
