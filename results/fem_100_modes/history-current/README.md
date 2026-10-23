# Compact search history for wide FEM (#147)

The existing public `keep_history=False` improves these wide FEM solves without
changing rank, convergence or completeness checks. It retains all current Ritz
vectors X but packs the R/P search directions by the current active residuals.
For block width b and active bucket width q, the projected width becomes b+2 q
instead of 2 b+q. Previously inactive rows can reactivate. Retaining more history
can reduce outer iterations, but its larger dense projected solves cost more.

These are default graph-policy controls on L40 / driver 570.158.01, Xeon 8362,
four threads on cores 8–11, 2026-10-23, pinned Warp/FSAI. The 34,560-DOF clamped
consistent-mass beam uses FSAI32, FP32 CG25 at relative tolerance 0.01, Warp outer
tolerance 1e-11, and widths 104/204 for 100/200 requested modes. All other settings
are identical within each pair, including inputs, starts, references, core
source and dependency hashes. The retained controls are reused from the
[production symmetry qualification](../../projected_symmetry/production/README.md).

| Modes | History | Setup s | Capture s | First-solve compute s | Warm median s | Outer iterations |
|---|---|---:|---:|---:|---:|---|
|100|retained|0.4044|1.6014|5.3172|3.0296|35, 35, 35|
|100|compact|0.4166|1.4113|4.6223|2.6800|54, 55, 55|
|200|retained|0.6112|4.8179|16.6859|10.4564|35, 36, 35|
|200|compact|0.5932|3.9231|13.8726|8.8615|57, 57, 57|

Replay improves **11.5% / 15.3%**. First-solve compute includes independent
preconditioner/solver setup, capture and the validated first captured solve.
It excludes mesh/reference assembly, imports and host validation; kernels are
cached. Setup/capture are single observations per fresh process, replay medians
use three solves. These are not whole-process wall times.

All 16 first/later outputs across the four workers pass: eight new compact-policy
outputs and eight reused retained controls. Worst original normalized residuals
are 8.98e-12 / 9.97e-12 for the compact 100/200 solves, with requested-subspace
errors 4.41e-11 / 5.68e-11. Native status/count/stopping, spectrum, mass orthogonality
and requested-space checks pass. Occupancy records are clean; graph DOT shows
no host nodes and only device-to-device copies. Raw JSON retains each sample,
arguments, quality checks, graph summaries and identities.

[Fresh matched-preconditioner CuPy comparisons](../history-cupy/README.md) and
[free/disconnected/heterogeneous controls](../history-families/README.md) qualify
the wider setting. This does not replace Anvil's retained-history recipe: its
sparse work/iteration tradeoff differs. No default or core algorithm is changed.

Reproduce with `benchmarks.experimental.compare_symmetry_api --symmetry default`
and the exact options in each occupancy record. Omitting
`--active-search-keep-history` selects the compact policy; include it for the
retained control. This is separate from `compact_projected_graphs`, which is
false throughout these workers.
