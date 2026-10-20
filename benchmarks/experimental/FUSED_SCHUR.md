# Cooperative QR loop fusion (#14, #110)

The prototype keeps the complete bounded shifted QR loop in a single Warp
block, with the Hessenberg matrix in shared memory and Schur vectors in global
memory. It preserves the prior deflation threshold, exceptional shifts,
iteration budget, right-vector extraction and original-equation final checks.

The first prototype passes 21 projected tests on pinned Warp and all 35
component timing samples. Saved 50 x 50 projections improve from about 18.0 ms to
13.7 ms. However, Warp 1.15 rejects its 2 D vector-tile scalar indexing (#110).
The corrected prototype uses a flat shared alias for scalar accesses, while
retaining row/column views for cooperative updates. Stock 1.15 passes 21 checks;
the common old/new suite passes 41 on pinned Warp in 20.04 s.
[Initial component records](../../results/fused-schur-components-initial.json),
[occupancy](../../results/fused-schur-components-initial.occupancy.json),
[compatibility failure](../../results/performance_followup/fused-schur-stock-failure.json).

After the correction, both complete #35 fixtures pass all 20 requested pairs
on each of three synchronized warm replays, with the original tolerances,
widths and budgets. Direct SR median is 2.889955 s (80–81 cycles), and cuDSS
shift-invert is 1.161794 s (fixed 30). Earlier accepted tiled medians were 3.216071 s
and 1.287696 s. These sequential measurements suggest roughly 10% improvement;
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


## Public integration validation

The public general solver now selects fused QR for 16–64-vector projections
when the GPU has sufficient shared memory (a conservative 96 KiB for padded 64,
32 KiB for smaller widths). The existing tiled path remains the lower-capacity
fallback, and the existing serial selection rules remain otherwise unchanged.
The shared Hessenberg/pre/postprocessing code is reused; fused construction
no longer creates an unused conditional QR graph. No eigensolver selection,
restart or stopping policy is changed.

The complete pinned suite passes **1,213 tests** (473.32 s); the stock Warp 1.15
focused suite passes **151** (37.10 s), including the shared-memory fallback.
The clean wheel contains 25 core modules byte-identical to source and excludes
benchmarks/tests/examples. Python lint and formatting pass.
[Validation](../../results/performance_followup/fused-schur-validation.json),
[wheel](../../results/performance_followup/fused-schur-wheel.json).

After integration, all 35 component timing samples again pass the independent
gates with clean observed occupancy. Saved direct and shifted projections take
13.731 and 13.743 ms, versus 17.992 and 18.011 ms for the tiled path. Random 64 takes
26.054 ms versus 31.780 ms. These are component results, not full sparse-solve
speedups. [Final component records](../../results/fused-schur-components-final.json),
[occupancy](../../results/fused-schur-components-final.occupancy.json).


The original 20-case LM inventory was repeated with unchanged k, ncv, tolerance
and Spectra start vectors, three replays per case. The same 11 cases pass and
the same 9 fail as before; the fused projection introduces no new failing case.
This retains the full requested-set gates and does not close #104. No failed
row is used as a performance win.
[Full inventory](../../results/fused-schur-general-selection.json).
