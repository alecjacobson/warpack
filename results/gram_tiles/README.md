# Gram output tile investigation (#180)

Benchmark-only FP64 Gram alternatives; **no installed default change**. The first component screen passes every output. Complete-eigensolve qualification is required before choosing an installed path.

L40 / Xeon 8362, pinned Warp 1.18 overlay, four CPU threads, reduction chunk 4096. All kernels use Warp FP64 arithmetic. No vendor matrix product is called. Setup/allocation, compilation, capture and CPU reference computation are excluded from component timing. First replay and five measured replays are recorded and checked separately. Every configuration uses identical strided input and output views.

| Physical dimension | Gram shape | Installed 8×8/128 ms | Candidate 16×16/128 ms | Component speed ratio |
|---:|---|---:|---:|---:|
| 34560 | 204×204 (symmetric) | 3.107 | 1.439 | 2.16× |
| 34560 | 408×408 (symmetric) | 11.331 | 5.259 | 2.15× |
| 34560 | 612×612 (symmetric) | 25.446 | 11.339 | 2.24× |
| 34560 | 408×204 (rectangular) | 11.278 | 5.010 | 2.25× |
| 34560 | 120×204 (rectangular) | 3.371 | 1.656 | 2.04× |
| 34560 | 24×204 (rectangular) | 0.714 | 0.549 | 1.30× |
| 34560 | 8×204 (rectangular) | 0.384 | 0.361 | 1.06× |
| 70656 | 120×120 (symmetric) | 2.196 | 1.437 | 1.53× |
| 70656 | 240×240 (symmetric) | 8.015 | 3.689 | 2.17× |

This table is a **component screen**, not an eigensolver speedup. The complete source manifest, all ten configurations (installed plus 8/16/32 tiles × 64/128/256 threads), every measured time and first-use time are in [screen-a1.json](screen-a1.json). The near-matching8×8/128 candidate is an explicit control for the rolled-loop implementation. The 16×16/64 variant has similar performance on the wide shapes; 128 threads was chosen for the first complete-solve screen because it is the existing launch grouping and passed the coupled full-solve tests. The thin8×204 product gains little. The 32 tile is slower than 16 on this device, and 256 threads is often worse than 128. All slower measurements are retained.

All **540 outputs** (90 configurations × six replays) pass independent CPU product checks, source/tail preservation and graph audits. Maximum relative product error is 8.52e-16 against a 5e-13 gate. Paired input identities, current source/dependency hashes and external GPU occupancy were verified. Graphs contain no HOST nodes or host/device copies. See [execution log](screen-a1.log) and [occupancy](screen-a1.occupancy.json).

## Rejected initial implementation

The initial kernel used a constant-length `for` loop. With reduction chunk 256, the compiler unrolled eight iterations and reserved 155648 bytes of shared tile storage for tile 32, exceeding the 100864 bytes available to that kernel. Six stage/full-solve tests failed at launch; 24 passed. This affected only the new prototype. [Exact rejected source](initial/gram_tiles.py.txt) and [failed test log](initial/tests.log) are preserved. A rolled `while` loop keeps the same FP64 accumulation sequence while reusing tile storage. The revised candidate passes 30 pinned-Warp tests and 30 stock-Warp 1.15 tests, including partial/strided/changed/scaled/zero products, both complete mass-QR passes, cached images, changed coupled pencils/constraints and repeated modes. Eight driver tests verify explicit controls and report/state restoration after rejected comparisons.

## Reproduction

`python -m benchmarks.experimental.compare_gram_tiles --out NEW.json` reproduces the component protocol; use `--reverse` for reverse configuration order. `--chunk 256` screens short reduction chunks; the default is 4096. Complete comparisons use `python -m benchmarks.experimental.compare_gram_whole --gram-tile 0|16 --reference-cache CACHE --public-api ...`, with exact FEM arguments recorded in each output. Only verified CPU reference arrays are reused outside backend setup timers; every complete worker builds its own preconditioner and solver.
