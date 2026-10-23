# Guarded direct branch recording (#168)

The installed active-search path records every branch directly for exact built-in
`SparseOperator`, `BatchedSparseOperator` and `Identity` stiffness/mass applications
on CUDA devices supporting conditional graphs. It avoids redundant numerical
Rayleigh–Ritz warmup of every branch. Custom subclasses and instance application
overrides retain the existing eager per-width preparation. The initial eager
solve, pre-capture synchronization (#61), callback ordering and replay arithmetic
are unchanged. No public option or numerical tolerance changed.

On the L40, pinned Warp/FSAI runtime and four Xeon cores (2026-10-23):

| Problem | Capture s, legacy / default | First-solve compute s, legacy / default | Replay median s, legacy / default |
|---|---:|---:|---:|
| anvil | 0.8130 / 0.6046 | 3.2179 / 3.0187 | 1.6049 / 1.6123 |
| 100 | 3.9045 / 1.6526 | 7.4633 / 5.2402 | 2.9785 / 3.0282 |
| 200 | 18.2003 / 4.8253 | 29.8469 / 16.7008 | 10.4611 / 10.4583 |
| 200-compact | 12.6884 / 1.6233 | 24.1417 / 13.0861 | 10.7680 / 10.7604 |

100/200 denote the clamped consistent-mass 34,560-DOF fixture with block widths
104/204. Anvil has 283,857 DOFs, 20 elastic modes and width 24. The compact row
additionally enables `compact_projected_graphs=True`; other rows use its default
`False`. The legacy control forces only the old eager warmup via a private flag.

All eight fresh workers pass the first captured output and three measured
replays: **32 complete accepted solves**. Every output satisfies native stopping
and independent original-equation, spectrum, mass-orthogonality and requested
subspace gates. Within each pair the matrices, reference, starts, source hashes,
dependencies and graph DOT summaries match exactly. External occupancy is clean.
Captured transfers are device-to-device, with no host nodes. This is graph DOT
inspection, not a CUDA API trace.

First-solve compute = complete solver/preconditioner setup + capture + the
validated first captured solve. It excludes mesh assembly, reference solving,
imports and host validation; compiled kernels are cached. It is not total process
wall time. Setup/capture are single observations per independent worker; replay
medians use three solves. Raw reports retain separate phases and all samples.
The 200-mode first-compute reduction is about 44% by default. The replay graph
is unchanged, and small replay timing differences are measurement/iteration
variation; this change does not accelerate its numerical algorithm. No CuPy
ratio or replacement six-backend Anvil table is inferred from these controls.

## Validation

- Pinned runtime: **79 passed, 1 skip** in the affected production suite.
- Stock Warp: **60 passed, 21 upstream-feature skips** in the affected suite.
- **6 guard tests passed**, including the additional unsupported-conditional-graph
  fallback added after the pinned suite (five overlap, one additional case).
- The same direct-recording prototype passed **74 pinned / 54 stock** tests with
  separate initially empty kernel caches; [cold-cache evidence](../capture-only/README.md).
- A clean wheel contains exactly 39 core modules matching source. The isolated
  installed-wheel smoke passes changed-input replays while forbidding imports
  of benchmarks, examples, SciPy, CuPy and the optional cuDSS package.

Regressions verify that all branches are recorded, built-in operators avoid
eager smaller-branch solves, and custom stiffness/mass subclasses and instance
overrides still get required per-width preparation. Existing affected tests
cover changed mass/constraints, repeated and rigid modes, callbacks, nonfinite
recovery, compact graphs and graph residency. The earlier full-suite baseline
is not presented as a fresh full-suite run of this small preparation change.

## Reproduction

`python -m benchmarks.experimental.compare_preparation_api --preparation legacy|default`
takes the usual `compare_block` options; add `--compact-graphs` for the compact
row. Each raw report records the complete argument set, source identities and
verified upstream dependency revision. Run each variant in a fresh process,
use the same inputs and pinned runtime, and wrap it in `benchmarks.monitor_gpu`.
The provided logs and occupancy reports accompany each worker.
