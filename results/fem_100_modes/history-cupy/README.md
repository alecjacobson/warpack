# Fresh wide-FEM CuPy controls after history tuning (#147)

Four independent workers on L40, pinned Warp/FSAI, four Xeon 8362 cores, 2026-10-23.
Each requests 100 or 200 modes of the same 34,560-DOF clamped consistent-mass beam.
Warp uses widths 104/204; CuPy uses 100/200 and the corresponding prefix of the
same starting block. Both use the supplied Warp FSAI32/FP32 CG25 preconditioner.
This compares outer solvers with matched preconditioning, not native CuPy-only
performance. All solver/preconditioner arithmetic on the Warp side remains Warp.

| Requested modes | Warp history | Warp warm median s | CuPy warm median s | Worst original residual, Warp / CuPy |
|---|---|---:|---:|---:|
|100|retained|3.0325|2.9378|9.59e-12 / 2.45e-11|
|100|compact|2.6674|2.9116|8.83e-12 / 2.45e-11|
|200|retained|10.6343|3.6890|9.07e-12 / 3.88e-11|
|200|compact|8.9078|3.6678|9.90e-12 / 3.89e-11|

Compact history makes Warp about 8.4% faster at 100 modes in this fixture, but
**Warp still takes 2.43 times as long at 200 modes**. These are medians of three
samples, not universal speed ratios. CuPy 100-mode samples vary: 2.4241–3.0204 s
in the compact-policy worker. Exact samples remain in the raw reports.

Warp stops on the requested normalized residuals at 1e-11. CuPy must satisfy its
1e-10 absolute bound over its entire returned block. These native criteria and
achieved accuracies differ; common independent original-equation, eigenvalue,
orthogonality and requested-subspace gates pass for both. Worst requested-space
error is 5.60e-11 for Warp and 5.22e-10 for CuPy across these workers.

All 28 outputs pass: three measured solves per backend per worker, plus each
worker's first captured Warp solve. No failed or contended worker is omitted
from this group. Within each history-policy pair, input/start/reference, source,
version and dependency identities match exactly. Every occupancy record is clean.

This table reports **warm solves only**. Within each worker CuPy receives the
preconditioner already built for Warp; its adapter preparation is not independent
end-to-end CuPy setup. No setup-speed ratio is inferred. Independent Warp setup,
capture and first-solve costs are reported in the
[Warp-only controls](../history-current/README.md), and all individual phases
remain in the raw JSON here. No six-backend Anvil table row is replaced by these
narrower controls.

Use the exact command in each occupancy JSON to reproduce. The wrapper selects
current default projected symmetry recovery, with ordinary projected graphs.
Only `--active-search-keep-history` differs within each pair; CuPy widths and
native tolerances remain fixed. #147 remains open for the 200-mode gap.
