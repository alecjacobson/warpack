# Fresh Anvil comparison: cuDSS 0.8 versus 1.0

Same 283,857-DOF problem, 26 modes, initial vector, reference, shift, FP64
Krylov settings, source content and pinned Warp/FSAI environment. Each version
owns three independent fresh-process setups and 15 measured replays. The order
alternates across rounds. All 30 measured and six first/warmup outputs pass
original physical and native convergence gates; occupancy is clean. No matching
or pivot override is needed for this shifted SPD matrix.

| Loaded cuDSS runtime (wheel) | Setup median (s) | Warm median (s) | Maximum original residual |
|---|---:|---:|---:|
| 0.8.0 (0.8.0.10) | 2.3804 | 0.4655 | 3.072e-09 |
| 1.0.0 (1.0.0.35) | 2.0922 | 0.4761 | 2.941e-09 |

The candidate reduces setup by 12.1%, while replay is **2.3% slower**. This is
not a blanket speedup. Solve batch-median spread is below 0.1% for each version.
The 0.8 first setup was 2.4621 s versus subsequent 2.3759/2.3804 s; all observations
are retained. Setup includes factor construction, outer workspaces and graph
capture. Assembly, mass normalization, imports, cached Warp compilation,
independent reference loading and validation are excluded. No cuDSS JIT disk
cache is enabled: candidate analysis includes any per-process NVRTC compilation.
These measurements do not isolate compilation from factorization or certify
cold-cache performance on another machine. Loaded binary/runtime identity is
queried after measurement so this audit does not preload the backend.

## Complete eigensolver graph and launch audit

Both versions also pass all 26 modes using the original start and two fresh
random starts (seeds 11 and 42), plus validation after three queued replays.

| Full captured graph | 0.8 | 1.0 |
|---|---:|---:|
| HtoD nodes | 128 | 0 |
| DtoD nodes | 198 | 134 |
| DtoH nodes | 0 | 0 |
| HOST callback nodes | 0 | 0 |

CUPTI traces show three graph-launch calls and no synchronization API or
numerical readback inside the warm enqueue region. An asynchronous allocator
cleanup also appears in both traces; this is not a claim that graph launch is
the only host API call. The explicit caller synchronization is recorded in a
separate phase. Instrumented traces are not used for performance claims. Graph
nodes/payloads are inspected read-only, and graphs are released before factors.

The supported dependency remains 0.8.0.10. Symmetric-indefinite 1.0 defaults need
an explicit capture-compatible pivot policy, and broader indefinite/KKT accuracy,
strict-pencil controls, JIT-cache behavior and deployment remain open in #189.
The current README table continues to describe its explicitly identified
supported-environment snapshot; it has not been partially replaced by these
candidate measurements.

[All setup/replay samples and checks](anvil/qualified-summary.json),
[0.8 graph/API audit](anvil/graph-0.8.0.json),
[1.0 graph/API audit](anvil/graph-1.0.0.json).
Source launchers, actual binary/binding hashes and occupancy records are retained
in `anvil/`. Environment: L40, driver 570.158.01, CPU affinity 8–11, four
OpenMP/OpenBLAS threads. Numerical source remains unchanged from the loader
integration c47a774; the profiler-only addition is identified by source hashes.
