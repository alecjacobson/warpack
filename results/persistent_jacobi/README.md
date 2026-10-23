# Persistent projected Jacobi prototype (#159)

This benchmark-only prototype keeps the global Jacobi solver's pair schedule,
right-then-left update arithmetic, 2e-15 convergence threshold and 30-sweep
bound, but executes the rounds in one CUDA block. Matrices and rotations live
in caller-supplied global workspaces. Tile collective barriers order dependent
global writes between rounds; the same tested Warp synchronization dependency
is documented in the [global-factor study](../global_cholesky/README.md).

Production dispatch is unchanged. A single block may lose bandwidth and
parallelism at large projected dimensions, so no speedup is assumed.

Eighteen pinned-overlay tests pass: even sizes 2,48,104,128,208,312,512,
indefinite random and repeated-eigenvalue spectra, scales 1e-80 through 1e80,
strided input/output/workspace, changed-input graph replay, independent SciPy
spectra, original equations, orthogonality, nonfinite/zero input and recovery,
and rejection of unsupported dimensions. The first development test attempt
failed compilation because Warp disallows tuple assignment to array elements;
the corrected kernel assigns each element separately. No numerical comparison
was run from that failed attempt.

`benchmarks.experimental.persistent_jacobi_compare` compares the existing
multi-kernel global solver against this prototype with exact input hashes,
validated warmup and three measured repeats per dimension, and graph audits.
The isolated comparison rejects this prototype for production; complete FEM
timing is unnecessary for a candidate that loses at every tested dimension.

## Rejected isolated performance control

L40, pinned Warp overlay, three measured replays plus a separately validated
warmup per dimension/backend. Exact input identities match within every pair.
Both backends use the same sweep counts and pass independent spectrum,
equation and orthogonality checks after warmup and every measured replay.
External occupancy is uncontended. Medians are milliseconds per complete
projected eigensolve, including device input reset and excluding setup/capture.

| Dimension | Existing global | Persistent single block |
| ---: | ---: | ---: |
| 48 | 1.645 | 2.319 |
| 72 | 2.533 | 7.792 |
| 104 | 4.196 | 24.645 |
| 128 | 5.275 | 45.133 |
| 208 | 11.543 | 206.001 |
| 312 | 22.027 | 691.664 |
| 512 | 59.242 | 3441.095 |

The persistent kernel is slower throughout, from 1.4× at 48 to 58× at 512.
Removing launches does not compensate for concentrating all global-memory and
FP64 work in one block. Equal sweep counts exclude a change in convergence
work as the explanation; detailed bandwidth/throughput attribution would need
additional instrumentation. No whole-FEM speedup or production change is claimed.

[Raw checks/timings/graph audits](isolated/jacobi-a1.json),
[occupancy](isolated/jacobi-a1.occupancy.json),
[worker log](isolated/jacobi-a1.log).
The broader projected-work gap remains tracked by #147; this particular approach
is rejected. The production global-workspace **Cholesky** optimization (#155)
is separate and retains its qualified benefit.
