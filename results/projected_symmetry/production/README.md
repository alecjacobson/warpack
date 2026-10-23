# Production projected symmetry recovery (#170)

The installed global projected Jacobi solver restores symmetry after every
sweep, before its existing convergence test. The ordinary and compact graph
policies share the same sweep finalization. The FP64 rotation order, 30-sweep
bound and 2e-15 diagonal-scaled stopping criterion remain unchanged. Small
shared-memory and odd-dimensional paths keep their existing implementation.

This fixes the recorded repeated/null-spectrum failure without doubling the
sweep budget. Both production graph policies converge on the shifted/rescaled
612-dimensional regression under the original stopping criterion. It adds one
Warp kernel per sweep and no solver buffers, host convergence decisions or
readbacks. No public option or requested tolerance changed.

## Complete same-source controls

On the L40, pinned Warp/FSAI runtime and four Xeon cores, 2026-10-23:

| Problem | Capture s, legacy / default | First-solve compute s, legacy / default | Warm replay median s, legacy / default |
|---|---:|---:|---:|
| anvil | 0.6185 / 0.6131 | 3.0107 / 3.0276 | 1.6100 / 1.6188 |
| 100 | 1.6061 / 1.6014 | 5.1946 / 5.3172 | 3.0235 / 3.0296 |
| 200 | 4.7691 / 4.8179 | 16.6535 / 16.6859 | 10.4695 / 10.4564 |
| 200-compact | 1.6548 / 1.6127 | 12.9446 / 12.8825 | 10.7602 / 10.7719 |

All eight fresh workers pass the first captured solve and three subsequent
replays: **32 complete accepted solves**. Native counts/status/stopping and
independent original physical residuals, eigenvalues, mass orthogonality and
requested subspaces pass. Within each pair matrices, references, starts, source
hashes and dependencies match exactly; external occupancy is clean. Graph DOT
inspection shows device-to-device transfers and no host nodes (not a CUDA API
trace). The extra restoration kernels intentionally change graph summaries.

The legacy control only disables the new restoration through a private flag.
The compact pair additionally enables `compact_projected_graphs=True`; other
rows use its default `False`. This is an accuracy/convergence fix, not a claimed
universal speedup. Warm differences are small and iteration trajectories vary.

First-solve compute = solver/preconditioner setup + capture + validated first
captured solve. It excludes mesh/reference work, imports and host validation,
with cached compiled kernels. It is not total process wall time. Setup/capture
are single observations per fresh worker; replay medians use three solves. All
separate phases and samples are in the raw reports. These rows do not replace
the six-backend Anvil table or establish a new CuPy ratio.

## Validation and review

[Current validation](../VALIDATION.md): **2,445 full-suite passes / 11 skips**;
171 affected tests pass on the pinned runtime; 168 pass on stock Warp with three
expected feature skips. A clean exact-source wheel and isolated changed-input
smoke pass with non-core/reference imports forbidden.

Review checks: each off-diagonal pair has one writer; the next kernel observes
the symmetric matrix in stream order. The original rotation and stopping loops
remain device controlled. Compact and ordinary round policies use identical
finalization and retain exact-result comparisons. Initial eager preparation,
pre-capture synchronization and callback order are unchanged. The private legacy
flag exists only for controls and is not a new public configuration.

## Reproduce

`python -m benchmarks.experimental.compare_symmetry_api --symmetry legacy|default`
accepts the usual public `compare_block` options; add `--compact-graphs` for
the compact pair. Each raw report retains its complete arguments, exact source
identities, dependency manifest and accuracy checks. Use fresh processes and
`benchmarks.monitor_gpu` with the pinned runtime.
