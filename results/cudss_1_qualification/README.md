# cuDSS 1.0 qualification in progress

Issue [#189](https://github.com/alecjacobson/warpack/issues/189) remains open.
The supported NVIDIA dependency is still **0.8.0.10**. Candidate **1.0.0.35**
is installed separately; the loaded runtime reports 1.0.0 and its binary SHA256
is recorded in `cudss1-library-identity.json`. Both versions use the same pinned
binding `3f2091fb1ae156609c38122dbb0a1d3c83913f74` and unchanged Warpack arithmetic.

## Capture compatibility

The initial run with 1.0 defaults passed two SPD tests, then failed
`test_indefinite_factor_first_capture[symmetric-False]` at capture with
`CUDSS_STATUS_EXECUTION_FAILED`. The failing log is retained.

NVIDIA documents that [AUTO pivot selection](https://docs.nvidia.com/cuda/cudss/latest/types.html#c.cudssPivotType_t)
may use Bunch–Kaufman for symmetric/Hermitian indefinite matrices; its solves
are synchronous and not stream-capturable. An isolated diagnostic explicitly
sets `CUDSS_CONFIG_PIVOT_TYPE=7` to `CUDSS_PIVOT_DIAGONAL=4` before analysis for
those matrix types. This retains the old diagonal-pivot strategy, with its
accuracy limitations, rather than claiming Bunch–Kaufman can be captured.
General/SPD policy and matching settings are otherwise unchanged. This control
is **not installed in the solver** and is not a silently chosen fallback.

| Candidate control | Result |
|---|---|
| Explicit diagonal policy: capture preparation, indefinite refinement, generalized transforms, complex GMRES refresh | 53 passed, 10.89 s |
| Complex shift/connection modes, generalized and damped pencils | 58 passed, 73.93 s |
| Read-only graph probes, 0.8 default policy | 6 cases, 18 changed-RHS outputs passed |
| Read-only graph probes, 1.0 explicit diagonal policy | 6 cases, 18 changed-RHS outputs passed |

The first test group includes 16 complex **GMRES** refresh tests, which are useful
regressions but do not exercise cuDSS. The second group includes actual cuDSS
complex realification and connection-mode solves as well as pure-Warp controls.
Passing these small cases does not qualify large FEM/KKT matrices or all pivot
configurations. Tests preserve original accuracy and native-convergence gates.

## Vendor metadata copies

The probes use 64-by-64 SPD, symmetric-indefinite and general matrices, each
with one and three RHS. Every case replays three changed RHS, checks the original
system against an independent dense solve, and verifies input preservation.

| Captured apply | cuDSS 0.8 HtoD nodes | cuDSS 1.0 HtoD nodes |
|---|---:|---:|
| SPD, no residual correction | 2 | 0 |
| Symmetric indefinite, one Warp residual correction | 4 | 0 |
| General, one Warp residual correction | 4 | 0 |

Each 0.8 node copies eight bytes containing `[0, 63]`; the corrected inverses
call the sparse solve twice, hence four nodes. Both versions have zero HOST
callback nodes and zero DtoH nodes. Largest relative original-system residuals
are about 2.1e-16. Graphs and copy payloads are inspected read-only while their
owners remain alive; no private graph rewrite is applied.

This is concrete progress on [#29](https://github.com/alecjacobson/warpack/issues/29),
not a claim that the currently supported 0.8 backend has changed. Full FEM graph
and enqueue audits, large-matrix accuracy/performance, explicit public policy,
JIT/cache setup accounting, and the strict pencil floor remain to be qualified.
The source launchers are retained as `.py.txt` with raw JSON/log evidence.

[Fresh Anvil timing and complete-graph qualification](ANVIL.md): candidate setup improves, warm replay is slightly slower, and all captured host/device transfer nodes disappear for this SPD problem. Broader qualification remains open.
