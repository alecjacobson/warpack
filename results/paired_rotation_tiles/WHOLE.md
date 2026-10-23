# Whole-solver paired-rotation pilot (#186)

All 48 first/measured outputs pass the unchanged original residual, requested spectrum/subspace, mass orthogonality, nullity and native status/count/stopping gates. Each worker constructs its own solver and preconditioner; only verified CPU references are cached outside timing. Input/source/dependency/reference identities match within every problem, occupancy records are clean, and graphs contain no HOST or non-device-to-device copies. The installed library is unchanged.

L40 / four Xeon 8362 cores / pinned Warp 1.18 overlay. Seconds below are three-replay medians. Setup and capture are single cached-compilation observations, not stability distributions. First-use compute includes setup, capture/eager preparation and first captured solve. Assembly, imports, references and validation are outside these timers.

| Problem | Rotation | Setup | Capture | First-use compute | Warm median | Change from original |
|---|---|---:|---:|---:|---:|---:|
| 200 | original | 0.6393 | 3.9246 | 11.3975 | 6.3931 | +0.00% |
| 200 | r8k8 | 0.6342 | 3.9529 | 11.1689 | 6.1170 | -4.32% |
| 200 | r8k32 | 0.6419 | 3.9342 | 11.1242 | 6.1170 | -4.32% |
| 200 | r16k32t256 | 0.6448 | 3.8581 | 11.0613 | 6.0655 | -5.13% |
| 100 | original | 0.4367 | 1.4145 | 4.0021 | 2.0353 | +0.00% |
| 100 | r8k8 | 0.4300 | 1.4838 | 4.0150 | 1.9827 | -2.59% |
| 100 | r8k32 | 0.4516 | 1.4965 | 4.0559 | 1.9819 | -2.62% |
| 100 | r16k32t256 | 0.4330 | 1.4860 | 4.0205 | 1.9990 | -1.78% |
| anvil | original | 0.7916 | 0.6158 | 2.9639 | 1.6056 | +0.00% |
| anvil | r8k8 | 0.8009 | 0.6934 | 3.0900 | 1.5379 | -4.21% |
| anvil | r8k32 | 0.7948 | 0.7058 | 3.0505 | 1.5430 | -3.90% |
| anvil | r16k32t256 | 0.7930 | 0.6971 | 3.0404 | 1.5515 | -3.37% |

`r8k8` = 8×32 output / inner8 / 128 threads; `r8k32` = 8×32 / inner32 / 128; `r16k32t256` = 16×32 / inner32 / 256. Original = 4×32 / inner8 / 128. All remain FP64.

The predeclared primary `r8k8` improves medians by 2.6–4.3%. Alternatives do not consistently improve the complete solve: the 16-row variant is fastest here on 200 modes but slower than the primary on 100 modes and Anvil. Retain `r8k8` for broader controls because it keeps the original inner tile and is consistently competitive. These are pilot observations, not a promoted dispatch or a new README timing table. Iteration differences remain visible in `whole-qualified.json`, including Anvil’s 16/17-step variation.

69 tests pass on both the pinned overlay (23.59s) and stock Warp 1.15 (24.17s). In addition to product/view tests, six complete coupled-mass regressions cover changed K/M/constraints, repeated eigenvalues and both history modes on captured replay; sixteen CPU driver tests cover explicit selection, compact graphs and restoration on rejected comparisons.

Next, run the primary before original in fresh reversed-order 200/100/Anvil controls and cover compact graphs, retained history, free-body nullspaces, disconnected components and heterogeneous FEM. All outcomes must be retained. Installed qualification follows only after those controls.

Raw worker reports, logs and occupancy are in [whole](whole/); [qualified summary](whole-qualified.json) records every sample, iteration count and error maximum.
