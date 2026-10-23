# Block Jacobi projected-solver experiment (#169)

This experiment remains outside the library. It diagonalizes disjoint small
panels with FP64 Warp shared-memory kernels and applies block rotations to the
projected matrix and basis. All work is Warp; no vendor dense solver is used.
The local scalar rotations follow the established projected solver's arithmetic.

## Rejected standalone variant

Full local diagonalization (up to 30 local sweeps per panel) accumulates too much
orthogonality error at large dimensions. On a repeated/rank-deficient 416-square
fixture it also exhausts the 30 outer sweeps with residuals above the unchanged
1e-11 gate. The initial test run stops at 26 passes and one failure; detailed
follow-up diagnostics cover block sizes 4/8/16 and local budgets 1/2/4/30.

`components-a1.json` retains the rejected component worker, including its first
and three subsequent outputs per row. Its occupancy record is ineligible because
numerical rejection produces exit code 1; no accepted whole-worker speed claim
is based on it. Exact source snapshots in `rejected-source/` match the two
experimental source hashes in that report. These are evidence of a failed
candidate, not library defaults or a replacement for the original accuracy gate.

## Block initialization plus canonical refinement

`BlockJacobiPolish` instead performs a fixed number of block-ordered FP64 sweeps,
then runs the installed scalar cyclic Jacobi convergence loop and composes the
bases with a Warp matrix product. The `sweeps` diagnostic counts refinement
sweeps; the additional initializer count is explicitly labeled in each backend.
The normal bounded refinement and 1e-11 component gates remain unchanged.

**53 tests pass on pinned Warp and 53 on stock Warp.** The pinned runs comprise
45 initial cases plus eight alternate-budget/boundary cases. They cover
indefinite and repeated/rank-deficient matrices, scales 1e-80..1e80, shifted
changed-input replay, strided outputs, padded odd dimensions, nonfinite recovery,
block sizes 4/8/16, initializer budgets 1/2/4, and dimensions up to 1024. No
production dispatch or public API changed.

Fresh L40/four-Xeon-core component controls (2026-10-23), same inputs and source,
three warm replays per row with a separately validated first output:

| Indefinite dimension | Global median ms | Four block sweeps + refinement ms |
|---|---:|---:|
| 104 | 4.186 | 4.546 |
| 208 | 11.594 | 11.806 |
| 416 | 38.889 | 34.217 |
| 612 | 83.183 | 78.616 |

All 16 indefinite rows (baseline and three initializer budgets) pass every gate
and the worker has clean external occupancy. Setup, eager preparation, capture,
first output, graph audit, input identity and dependency hashes are retained in
`polish-indefinite-a1.json`; this table is a component result, not a FEM speedup.

The separate rank-deficient worker is **rejected**: the installed global solver
at dimension 612 exhausts 30 sweeps with original residual 1.14203e-11. Its
spectrum and orthogonality pass, and all initializer candidates pass this matrix.
[Issue #170](https://github.com/alecjacobson/warpack/issues/170) tracks this new
projected-component limitation; it does not establish a wrong accepted public
FEM solve. No speed ratio uses the rejected worker. All evidence is retained in
`polish-rank_deficient-a1.json` and its log/occupancy report.

## Reproduce

```bash
python -m benchmarks.experimental.block_jacobi_compare \
  --polish --kind indefinite --sizes 104,208,416,612 --out <fresh-report.json>
python -m benchmarks.experimental.block_jacobi_compare \
  --polish --kind rank_deficient --sizes 104,208,416,612 --out <fresh-report.json>
```

Use `benchmarks.monitor_gpu` and the pinned runtime for comparable measurements.
Omit `--polish` to reproduce the rejected standalone policy. Whole-FEM controls use `compare_block_jacobi`, selecting existing even projected
sizes 208..1024, four initializer sweeps and block size eight. Smaller paths,
including the Anvil projected sizes, stay unchanged.

Primary background: [parallel block Jacobi algorithms](https://www.sciencedirect.com/science/article/pii/002437958690176X)
and [combinatorial ordering for parallel block Jacobi](https://doi.org/10.14495/jsiaml.13.56).
This prototype uses fixed cyclic pairs; it does not implement the latter paper's
permutation strategy or Saad's distinct Riccati-based subspace rotations.


## Complete FEM result: no promotion

All six fresh workers in `fem/` pass the first output and three subsequent
replays: **24 complete accepted solves**. Native stopping, original physical
residuals, eigenvalues, mass orthogonality and requested subspaces pass. Within
each pair input/source/dependency identities match, and external occupancy is
clean. The Anvil pair has identical graph summaries and no eligible projected
sizes, as expected.

| Problem | Capture s, default / candidate | First-solve compute s, default / candidate | Warm replay median s, default / candidate |
|---|---:|---:|---:|
| 100 modes | 1.6109 / 2.1289 | 5.1962 / 5.7264 | 2.9809 / 3.0101 |
| 200 modes | 4.6936 / 8.1294 | 16.5545 / 20.5932 | 10.6462 / 10.6519 |
| Anvil, unchanged path | 0.6096 / 0.6120 | 2.9965 / 3.0596 | 1.6572 / 1.6628 |

The candidate offers no useful warm-replay gain and substantially increases
capture and first-solve work. It is **not promoted**. Component gains on selected
dense matrices did not translate into a faster complete FEM mode solve.
[Issue #169](https://github.com/alecjacobson/warpack/issues/169) is closed as not
planned for this implementation; the broader wide-mode gap remains in #147.

First-solve compute includes solver/preconditioner setup, capture and a validated
first captured solve, excluding mesh/reference work, host validation and imports.
Compilation is cached; setup/capture are single observations per fresh worker.
This is not total process wall time and does not replace the six-backend Anvil
table or establish any new CuPy speed ratio.
