# Disposable-history warmup control (#168)

Clearing setup-only search history preserves accuracy and graph structure, but removes only a small part of preparation cost. This candidate is retained as experimental evidence while direct recording of all search branches is evaluated. Installed warmup remains unchanged.

All eight workers pass independent first-output and three later replay gates (32 complete solves), paired input/source/dependency identities, and external occupancy checks. Graph DOT summaries are identical within every pair.

| Problem | Capture s, current / zero-history | First-solve compute s, current / zero-history | Replay median s, current / zero-history |
|---|---:|---:|---:|
| anvil | 0.8127 / 0.8168 | 3.2160 / 3.2213 | 1.6053 / 1.6132 |
| 100 | 3.8928 / 3.7951 | 7.3375 / 7.2854 | 2.9754 / 3.0315 |
| 200 | 18.2275 / 17.6638 | 30.0441 / 29.3127 | 10.4806 / 10.4886 |
| 200-compact | 12.6326 / 12.2794 | 24.1047 / 23.5987 | 10.5705 / 10.7542 |

First-solve compute is setup + capture + the independently validated first captured solve, excluding mesh/reference work, initial imports and host validation. Setup/capture are single observations per fresh worker with cached compilation. All original per-phase scopes and replay samples remain in the JSON reports. These do not replace the six-backend Anvil table.

Candidate regression tests: 74 passed / 1 skip pinned; 54 passed / 21 upstream-feature skips stock. They run existing coupled-mass, repeated/rigid-space, changing-input, odd-width, nonfinite-recovery, FEM and compact-graph cases with the candidate installed by a test-only plugin. Initializer/finalizer tests now cover active search as well as the unbucketed solver, checking ordering and exactly one device callback per replay.

Most of the measured 11-second wide warmup remains after history clearing. The next candidate records all branches directly, using supported CUDA capture-time kernel loading while retaining the initial eager solve and pre-capture synchronization. It requires separate cold-cache, custom-operator compatibility and complete performance qualification.
