# Projection tile screen

Benchmark-only output tile variants retain FP64 arithmetic, inner tiles of eight, 32 output columns and 128 threads. The original installed kernel and generated four-row control are included. No numerical core or solver policy has changed.

All 128 first/measured outputs (eight shapes, four controls, four outputs each) pass independent CPU products, finite error gates, exact input/padding checks and captured graph audit. Loaded dependency/source identity and external occupancy checks pass. Component and changed coupled-metric/constraint/cluster replays: 22 tests pass on both pinned Warp and stock 1.15. Comparison-driver restoration/rejection controls: 16 pass.

Median milliseconds, three synchronized warm replays per component. Original output values are reset outside every timer.

| n:output:basis | Installed 4 | Generated 4 | Generated 8 | Generated 16 |
|---|---:|---:|---:|---:|
| 34560:408:204 | 10.012 | 10.011 | 5.053 | 10.078 |
| 34560:120:204 | 2.973 | 2.972 | 1.514 | 3.125 |
| 34560:24:204 | 0.618 | 0.617 | 0.323 | 0.839 |
| 34560:8:204 | 0.231 | 0.231 | 0.143 | 0.433 |
| 34560:208:104 | 2.583 | 2.582 | 1.319 | 2.548 |
| 34560:8:104 | 0.124 | 0.123 | 0.079 | 0.222 |
| 283857:48:24 | 1.183 | 1.182 | 0.650 | 1.148 |
| 283857:28:24 | 0.707 | 0.708 | 0.450 | 0.773 |

Eight output rows improve these component timings by about 1.57–1.98×; sixteen are often slower. All variants are retained. Whole-solver controls on 200 modes, 100 modes and Anvil are running; no whole-solve gain or production dispatch is claimed from this screen.

[Raw screen](screen/a1.json).
