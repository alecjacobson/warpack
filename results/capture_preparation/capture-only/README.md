# Direct branch recording candidate (#168)

Record every active-search branch in the final graph without an extra numerical solve of each branch beforehand. The initial eager solve and synchronization before capture remain. Conditional CUDA graph support requires a driver that also supports loading kernels during capture. This benchmark candidate is qualified on concrete library-owned operators; arbitrary custom operators require a conservative compatibility fallback before production integration.

All eight complete workers pass independent first-output and three subsequent replay gates (32 complete solves). Each pair has exact input/source/dependency identities and identical graph DOT summaries; external occupancy is clean.

| Problem | Capture s, current / direct recording | First-solve compute s, current / direct recording | Replay median s, current / direct recording |
|---|---:|---:|---:|
| anvil | 0.8147 / 0.6049 | 3.2366 / 3.0080 | 1.6648 / 1.6113 |
| 100 | 4.0045 / 1.6824 | 7.6108 / 5.1742 | 3.0329 / 2.9752 |
| 200 | 18.1287 / 4.8462 | 30.0048 / 16.4985 | 10.4684 / 10.4681 |
| 200-compact | 12.8357 / 1.6562 | 24.3061 / 12.9590 | 10.7605 / 10.7707 |

First-solve compute is setup + capture + the independently validated first captured output. It excludes host validation, mesh/reference work and initial imports; cached compilation is used where available. It is not total process wall time. Setup/capture are single observations per fresh worker. All separate phases and replay samples are retained in the raw reports.

For 200 modes, this reduces first-solve compute about 45% with default projected graphs and 47% with compact projected graphs. The replay graph and arithmetic are unchanged; small warmed-time differences are measurement/iteration variation, not an algorithmic replay speedup. This is separate from the six-backend Anvil table and contains no new CuPy ratio.

Cold-cache regression on the pinned runtime: **74 passed, 1 skip**, 144.82s including compilation into a new empty kernel cache. Stock Warp: **54 passed, 21 upstream-feature skips**, 93.52s using its separate previously empty version directory. Tests cover changed input/mass/constraints, repeated and rigid modes, odd projected sizes, nonfinite recovery, callbacks and graph residency. Cached earlier code is therefore not required for correct branch capture.

Compatibility review: custom operators may lazily prepare per-width storage before capture. Four baseline regressions demonstrate that existing eager preparation supports subclasses and instance overrides on either stiffness or mass. The production route must preserve that preparation for custom applications. The normal-owned-operator regression will also require every branch to be recorded without eager evaluation of small branches.
