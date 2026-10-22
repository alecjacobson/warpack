# Linear-storage CuPy CG adapter

Issue #146: the comparison adapter previously preallocated one upstream Warp
CG state for every active width. At width 104, that reserves 5,460 RHS slots.
The opt-in `CapturedPartitionedCG` instead reserves widths 1,2,4,8,16,32,64:
127 slots. A 104-RHS call runs 64+32+8 exact RHSs, with no zero padding. One
additional full-width input buffer preserves partially overlapping input/output
views. Numerical work remains unchanged upstream Warp CG and Warp kernels;
there is no private workspace rebinding or device allocation during replay.

Chunks stop independently. This can reduce scheduled masked work, but adds up
to seven graph launches per call, so the adapter is not assumed to improve
runtime. The existing whole-active-block adapter remains the default. Enable
the candidate with `--cupy-partitioned` in `benchmarks.experimental.compare_block`.
The choice and per-chunk native statistics are included in each raw report.

## Qualification

58 focused tests pass, covering FP32/FP64, both operator/FSAI layouts, changing
matrix values and active widths, zero and scaled RHSs, exact and shifted aliases,
shared factor/storage, native failed-RHS/iteration accounting, graph auditing,
and replay without numerical workspace allocation. The full 34,560-DOF,
100-mode consistent-mass FEM comparison also passes all original-spectrum,
subspace, orthogonality and method-native stopping checks for three replays.

The process-local Warp-pool used-memory delta falls from **10,429,234,172 bytes
(9.71 GiB)** for the exact-width adapter to **274,685,624 bytes (0.256 GiB)** for
partitioning, about **38× less**. These are live pool allocation deltas after
setup, not peak or total-context memory. They exclude driver/graph/non-pool
allocations; foreign process allocations do not contribute to this process's
pool counter. The baseline screening run was contended and contributes memory
and correctness evidence only, not a timing comparison.

[Exact-width screen](screening/exact-a1.json),
[partitioned screen](screening/partitioned-a1.json). The partitioned worker is
uncontended and accepted, with Warp median 3.7113 s and CuPy median 3.5529 s.
CuPy's individual replays are 3.5529, 3.7798 and 2.9021 s. This variability and
the different outer stopping contracts still matter: Warp targets 100 normalized
residuals at 1e-11; CuPy targets 104 absolute residuals at 1e-10. The worst
original residual is 9.51e-12 for Warp and 7.32e-11 for CuPy. These data do not
establish a speedup over the contended exact-width screening control.

For this partitioned worker, Warp preconditioner/solver setup and capture are
recorded separately in the raw result. CuPy adapter initialization/capture is
0.0530 s; matrix/start/adapter preparation together is 0.0876 s. This latter
stage reuses the already constructed FSAI/base CG and excludes their creation;
it is not a standalone end-to-end CuPy setup measurement. Each chunk graph's
initial upload/launch completes during adapter setup to avoid the known #61
first-upload interaction. There is no synchronization between chunks during
normal replay.

Use each occupancy JSON's exact command for reproduction. The
`fem_wide_report` acceptance gate remains applicable and records the selected
adapter in its configuration. Different source/configuration records require
separate reviewed baselines. The larger Warp/CuPy replay gap remains #147;
shrinking the comparator's workspace does not fix that separate solver issue.
