# Paired Ritz-rotation component screen (#186)

Benchmark-only prototype; no installed solver or default changed. Every first and measured pair is compared with independent FP64 CPU products, with exact checks of inputs, padding and view boundaries. Captured graphs have no host nodes or non-device-to-device copies. The external occupancy record is retained. Compilation, allocation, capture, reference products and validation are outside replay timing.

90 configurations (nine shapes, ten implementations), each with one first replay and three measured replays, passed: 360 paired calls, 720 output-error checks. An explicit post-run audit also confirmed every error is finite, nonnegative and strictly below 5e-13; the largest was 1.2013618737598922e-15. This audit matters because #188 identifies a NaN hole in the original aggregate acceptance expression. No actual error in this screen exploits that hole. Original as-run source hashes remain in the JSON; the subsequent validation fix must not backfill them.

The fastest component configuration for each shape is shown below. These are selected component minima, **not whole-solver speedups**. All ten configurations, including slower variants, remain in `screen-a1.json`. Whole-solver controls must establish accuracy and performance before promotion.

| n / block / search | Best rows:columns:inner:threads | Original ms | Candidate ms | Component ratio |
|---|---|---:|---:|---:|
| 34560:204:408 | 16:32:32:256 | 20.363 | 7.328 | 2.78 |
| 34560:204:120 | 8:32:32:128 | 9.011 | 4.061 | 2.22 |
| 34560:204:24 | 16:32:32:256 | 6.008 | 2.900 | 2.07 |
| 34560:204:8 | 8:32:8:128 | 5.500 | 2.826 | 1.95 |
| 34560:104:208 | 8:32:32:128 | 4.669 | 2.104 | 2.22 |
| 34560:104:8 | 8:32:8:128 | 1.507 | 0.774 | 1.95 |
| 283857:24:48 | 8:32:8:128 | 2.350 | 1.201 | 1.96 |
| 283857:24:28 | 8:32:32:128 | 1.788 | 0.812 | 2.20 |
| 70656:120:240 | 16:32:32:256 | 14.435 | 5.630 | 2.56 |

## Regression checks and stock-Warp compatibility (#187)

47 tests pass on both the pinned Warp 1.18 overlay and stock Warp 1.15. The tests include zero search width, strided partial tiles, changed-input graph replay, scales from 1e-100 to 1e100, exact padding/input preservation, both independent output products and restoration of the prototype launch wrapper on success/failure.

The initial stock run had nine validation failures after its output checks passed: Warp 1.15 cannot materialize an empty non-null sliced device view with `.numpy()`. `initial/` retains that failing test and log. Validation now reads each nonempty backing allocation and slices the resulting NumPy array, including empty slices. No numerical kernel, threshold or installed code changed; this resolves our test compatibility issue, not the upstream empty-view behavior. The passing rerun logs are alongside this report.

## Whole-solver plan

Predeclared primary candidate: 8×32 output, inner tile 8, 128 threads (preserves the original inner reduction geometry). Explicit alternatives: 8×32 / inner32 / 128 threads and 16×32 / inner32 / 256 threads. Compare each with the original on Anvil and the 100-/200-mode FEM workloads, checking the first result and all three measured results against unchanged physical and requested-spectrum gates. Keep every result. Broader/reversed-order and installed qualification follow only if whole-solver evidence warrants it.
